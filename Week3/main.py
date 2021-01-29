import os
import sys
import logging

import atari_wrappers
import hydra
import numpy as np
import omegaconf
import gym
from tensorboardX import SummaryWriter

from agent import DQNAgent

logger = logging.getLogger(__name__)

def flatten_dict(dd, separator ='_', prefix =''): 
    return { prefix + separator + k if prefix else k : v 
             for kk, vv in dd.items() 
             for k, v in flatten_dict(vv, separator, kk).items() 
             } if isinstance(dd, omegaconf.DictConfig) else { prefix : dd } 

@hydra.main(config_path='conf', config_name='config')
def main(cfg: omegaconf.DictConfig):

	# create the environment
	env = atari_wrappers.make_env(cfg.exp.env)
	env = gym.wrappers.Monitor(env, "recording/", force=True)
	obs = env.reset()

	# TensorBoard
	writer = SummaryWriter()
	writer.add_hparams(flatten_dict(cfg), {})
	logger.info('Hyperparams:', cfg)

	# create the agent
	agent = DQNAgent(env, device=cfg.train.device, summary_writer=writer, cfg=cfg)

	n_games = 0
	max_mean_40_reward = -sys.maxsize

	# Play MAX_N_GAMES games
	while n_games < cfg.train.max_episodes:
		# act greedly
		action = agent.act_eps_greedy(obs)

		# one step on the environment
		new_obs, reward, done, _ = env.step(action)

		# add the environment feedback to the agent
		agent.add_env_feedback(obs, action, new_obs, reward, done)

		# sample and optimize NB: the agent could wait to have enough memories
		agent.sample_and_optimize(cfg.train.batch_size)

		obs = new_obs
		if done:
			n_games += 1
			agent.print_info()
			agent.reset_stats()
			obs = env.reset()
			if agent.rewards:
				current_mean_40_reward = np.mean(agent.rewards[-40:])
				if current_mean_40_reward > max_mean_40_reward:
					agent.save_model(cfg.train.best_checkpoint)
	writer.close()

if __name__ == "__main__":
    main()
