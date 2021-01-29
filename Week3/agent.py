import numpy as np
from collections import namedtuple
import torch
import time
import logging

from central_control import CentralControl
from buffers import ReplayBuffer

logger = logging.getLogger(__file__)

class DQNAgent:
	'''
	Agent class. It control all the agent functionalities
	'''
	Memory = namedtuple('Memory', ['obs', 'action', 'new_obs', 'reward', 'done'], rename=False)

	def __init__(self, env, device, cfg, summary_writer=None):
		'''
		Agent initialization. It create the CentralControl that control all the low
		'''

		# The CentralControl is the 'brain' of the agent
		self.cc = CentralControl(env.observation_space.shape, env.action_space.n, cfg.rl.gamma, cfg.rl.n_multi_step, cfg.neural_net.double_dqn,
				cfg.neural_net.noisy_net, cfg.neural_net.dueling, device)

		self.cc.set_optimizer(cfg.train.learning_rate)

		self.birth_time = time.time()

		self.iter_update_target = cfg.replay.n_iter_update_target
		self.buffer_start_size = cfg.replay.buffer_start_size

		self.epsilon_start = cfg.rl.epsilon_start
		self.epsilon = cfg.rl.epsilon_start
		self.epsilon_decay = cfg.rl.epsilon_decay
		self.epsilon_final = cfg.rl.epsilon_final

		self.accumulated_loss = []
		self.device = device

		# initialize the replay buffer (i.e. the memory) of the agent
		self.replay_buffer = ReplayBuffer(cfg.replay.buffer_capacity, cfg.rl.n_multi_step, cfg.rl.gamma)
		self.summary_writer = summary_writer

		self.noisy_net = cfg.neural_net.noisy_net

		self.env = env

		self.total_reward = 0
		self.n_iter = 0
		self.n_games = 0
		self.ts_frame = 0
		self.ts = time.time()
		self.rewards = []

	def act(self, obs):
		'''
		Greedy action outputted by the NN in the CentralControl
		'''
		return self.cc.get_max_action(obs)

	def act_eps_greedy(self, obs):
		'''
		E-greedy action
		'''

		# In case of a noisy net, it takes a greedy action
		if self.noisy_net:
			return self.act(obs)

		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		else:
			return self.act(obs)

	def add_env_feedback(self, obs, action, new_obs, reward, done):
		'''
		Acquire a new feedback from the environment. The feedback is constituted by the new observation, the reward and the done boolean.
		'''

		# Create the new memory and update the buffer
		new_memory = self.Memory(obs=obs, action=action, new_obs=new_obs, reward=reward, done=done)
		self.replay_buffer.append(new_memory)

		# update the variables
		self.n_iter += 1
		# decrease epsilon
		self.epsilon = max(self.epsilon_final, self.epsilon_start - self.n_iter/self.epsilon_decay)
		self.total_reward += reward

	def sample_and_optimize(self, batch_size):
		'''
		Sample batch_size memories from the buffer and optimize them
		'''

		if len(self.replay_buffer) > self.buffer_start_size:
			# sample
			mini_batch = self.replay_buffer.sample(batch_size)
			# optimize
			l_loss = self.cc.optimize(mini_batch)
			self.accumulated_loss.append(l_loss)

		# update target NN
		if self.n_iter % self.iter_update_target == 0:
			self.cc.update_target()

	def reset_stats(self):
		'''
		Reset the agent's statistics
		'''
		self.rewards.append(self.total_reward)
		self.total_reward = 0
		self.accumulated_loss = []
		self.n_games += 1

	def save_model(self, model_path):
		checkpoint = {
			"episode": self.n_games,
			'optimizer': self.cc.optimizer.state_dict(),
			'network': self.cc.moving_nn.state_dict()
		}
		torch.save(checkpoint, model_path)

	def print_info(self):
		'''
		Print information about the agent
		'''
		fps = (self.n_iter-self.ts_frame)/(time.time()-self.ts)
		logger.info('%d %d rew:%d mean_rew:%.2f eps:%.2f, fps:%d, loss:%.4f' % (self.n_iter, self.n_games, self.total_reward, np.mean(self.rewards[-40:]), self.epsilon, fps, np.mean(self.accumulated_loss)))

		self.ts_frame = self.n_iter
		self.ts = time.time()

		if self.summary_writer != None:
			self.summary_writer.add_scalar('reward', self.total_reward, self.n_games)
			self.summary_writer.add_scalar('mean_reward', np.mean(self.rewards[-40:]), self.n_games)
			self.summary_writer.add_scalar('10_mean_reward', np.mean(self.rewards[-10:]), self.n_games)
			self.summary_writer.add_scalar('epsilon', self.epsilon, self.n_games)
			self.summary_writer.add_scalar('loss', np.mean(self.accumulated_loss), self.n_games)
