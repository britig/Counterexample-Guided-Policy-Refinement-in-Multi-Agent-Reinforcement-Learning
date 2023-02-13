# GPU implementation of the PPO algorithm

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from network import FeedForwardActorNN, FeedForwardCriticNN
import sys
from eval_policy import eval_policy
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PPO(object):
	def __init__(self, subpolicy,type, env, **hyperparameters):

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)
		
		# Extract environment information
		self.env = env
		self.obs_dim = env.env.env.env.observation_space[0].shape[0]
		self.subpolicy = subpolicy
		self.type = type
		actor_model_old = 'ppo_actor_updatedmultiwalker_v7_primary_walker0.pth'
		critic_model_old = 'ppo_criticmultiwalker_v72.pth'
		if self.discrete:
			self.act_dim = env.action_space.n
		else:
			self.act_dim = env.env.env.env.action_space[0].shape[0] #env.action_space.n #env.action_space.shape[0]

		# Initialize actor and critic networks
		self.actor = FeedForwardActorNN(self.obs_dim, self.act_dim,self.discrete) 
		self.actor.to(device)
		#print(f'model =========== {self.actor}')                 	# ALG STEP 1
		self.critic = FeedForwardCriticNN(self.obs_dim, 1)
		self.critic.to(device)
		#print(f'critic =========== {self.critic}') 

		# If training with the secondary objective start with the old policy as we want to minimize distance between the policies
		if self.type == 'secondary':
			self.actor.load_state_dict(torch.load(actor_model_old))
			self.critic.load_state_dict(torch.load(critic_model_old))

		# Initialize optimizers for actor and critic
		self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)
		
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var).to(device)
		self.obs_count = 0
		self.index_count = 0
		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
			'actor_network' : 0,	# Actor network
		}


	def select_action(self, state):
		state = torch.FloatTensor(state).to(device)
		return self.actor(state).cpu().data.numpy()


	def learn(self, env_name,failure_observations):
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {self.training_step} iterations")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		# Sample replay buffer
		while i_so_far < self.training_step:  
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout(failure_observations)

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far
			
			V = self.critic(batch_obs.to(device)).squeeze()
			
			# Advatage calculation
			A_k = batch_rtgs.to(device) - V
			# Normalize advantages
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			print(f'n update======={self.n_updates_per_iteration}')
			
			for _ in range(self.n_updates_per_iteration): 
				print(f'inside======={i_so_far}')
				
				V = self.critic(batch_obs.to(device)).squeeze()
				curr_log_probs = self.evaluate(batch_obs, batch_acts)
				
				# Calculate ratio
				ratios = torch.exp(curr_log_probs - batch_log_probs.detach().to(device))
				
				# Calculate surrogate losses
				surr_loss1 = ratios * A_k.detach()
				# clamp() performs clipping operation, binding arg1 between arg2, arg3
				surr_loss2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k.detach()
					
				critic_loss = F.mse_loss(V, batch_rtgs.to(device))
				
				self.critic_optimizer.zero_grad()
				critic_loss.backward(retain_graph=True)
				self.critic_optimizer.step()
				
				# Calculate actor loss using surr_losses
				# Loss is maximized using Stochastic gradient ascent
				actor_loss = -torch.min(surr_loss1, surr_loss2).mean()
				
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach().cpu().numpy())
				self.logger['actor_network'] = self.actor

			# Print a summary of our training so far
			self._log_summary()

			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				if self.subpolicy:
					torch.save(self.actor.state_dict(), './ppo_actor_subpolicy'+env_name+'_'+self.type+'.pth')
					torch.save(self.critic.state_dict(), './ppo_critic_subpolicy'+env_name+'_'+self.type+'.pth')
				else:
					torch.save(self.actor.state_dict(), './ppo_actor'+env_name+'.pth')
					torch.save(self.critic.state_dict(), './ppo_critic'+env_name+'.pth')

		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		return avg_ep_rews
				
	def action_log_probs(self, action_mean):
		
		action_mean = torch.FloatTensor(action_mean).to(device)
		# cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
		dist = MultivariateNormal(action_mean, self.cov_mat)
		action = dist.sample()
		action_logprob = dist.log_prob(action)
		
		return action_logprob
	
	def evaluate(self, batch_obs, batch_acts):
		
		# To obtain the most recent log probabilities
		mean = self.actor(batch_obs.to(device))
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts.to(device))
		
		return log_probs
	
	def compute_rtgs(self, batch_rews):
		batch_rtgs = []
		
		for ep_rews in reversed(batch_rews):
			discounted_reward = 0
			
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)
	
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
		return batch_rtgs
	
	def rollout(self, failure_observations):
		batch_obs = []			# Batch observations, dim = [no of timesteps per batch, dim of observation]
		batch_acts = []			# Batch actions, dim = [no of timesteps per batch, dim of action]
		batch_log_probs = []	# Log probs of each action, dim = [no of timesteps per batch]
		batch_rews = []			# Batch rewards, dim = [no of episodes, no of timesteps per episode]
		batch_rtgs = []			# Batch returns, dim = [no of timesteps per batch]
		batch_lens = []			# Episodic lengths in batch, dim = [no of episodes]

		ep_rews = []
		
		t = 0
		while t < self.timesteps_per_batch:
			ep_rews = []
			obs = self.env.reset()
			done = False

			#If subpolicy sample trajectories from failure observations 
			if self.subpolicy:
				env_name = self.env.metadata['name']
				#print(env_name)
				len_obs = len(failure_observations)
				index = self.obs_count%len_obs
				seed = failure_observations[index][2]
				self.env.env.env.env.seed(seed[0])
				self.env.reset()
				obs = failure_observations[index][0][0]
				spec_type = failure_observations[index][4]

				# Extract the walker number and failure parameter
				for j in range(len(failure_observations[index][3])):
					#print(failure_observations[index][3][j][0])
					self.env.env.env.env.walkers[failure_observations[index][3][j][1]].hull.angle = failure_observations[index][3][j][0]

				self.obs_count = self.obs_count+1
			else:
				# Reset the environment. Note that obs is short for observation. 
				obs = self.env.reset()
			
			done = False

			for ep_t in range(self.max_timesteps_per_episode):
				obs, rew, done, info = self.env.last()
				# Training with penalty + reward the primary objective
				# The penalty is different for each type of specification
				if self.subpolicy and  self.type == 'primary':
					if env_name == 'multiwalker_v7':
						# Calculate penalty from all types of specs
						if spec_type == 'local':
							#Add only if spec is negative
							if((0.7 - obs[0])<0):
								rew = rew + (0.7 - obs[0])/10
						# Penalty from distance
						#If there is a left neighbour then penalise for crash with left neighbour
						if spec_type == 'mutual':
							if(obs[24] != 0):
								if((obs[24]*-1 - 0.24)<0):
									rew =  rew + (obs[24]*-1 - 0.24)/10
							#If there is  right neighbour then penalise for crash with right neighbour
							if(obs[26] != 0):
								if((obs[26] - 0.24)<0):
									rew =  rew + (obs[26] - 0.24)/10
						#Penalty from package angle
						if spec_type == 'global':
							if((0.25 - obs[30])<0):
								rew = rew + (0.25 - obs[30])/10
					if env_name == 'waterworld_v3':
						if spec_type == 'global':
							pass

				#Training with penalty and minimizing distance to old policy secondary objective
				if self.subpolicy and env_name == 'multiwalker_v7' and self.type == 'secondary':
					if spec_type == 'local':
						if((0.7 - obs[0])<0):
							rew = (0.7 - obs[0])/10 
						if((0.7 - obs[0])>0):
							rew = rew
						else:
							rew = rew
					if spec_type == 'mutual':
						#If there is a left neighbour then penalise for crash with left neighbour
						if(obs[24] != 0):
							if((obs[24]*-1 - 0.24)<0):
								rew =  (obs[24]*-1 - 0.24)/10
							if((obs[24]*-1 - 0.24)>0):
								rew =  rew
							else:
								rew = rew
						#If there is  right neighbour then penalise for crash with right neighbour
						if(obs[26] != 0):
							if((obs[26] - 0.24)<0):
								rew =  (obs[26] - 0.24)/10
							if((obs[26] - 0.24)>0):
								rew =  rew
							else:
								rew = rew
					if spec_type == 'global':
						if((0.25 - obs[30])!=0):
							rew = (0.25 - obs[30])/10
						else:
							rew = 10
					
				# If the environment tells us the episode is terminated, break
				if done:
					break
				#print(f'obs ===== {obs} === rew ======= {rew} done ==== {done}')
				# If render is specified, render the environment
				if self.render:
					self.env.render()

				t += 1
				batch_obs.append(obs)
				
				action = self.select_action(np.array(obs))
				# action = self.noise.get_action(action, ep_t)
				log_prob = self.action_log_probs(action)
				self.env.step(action)
				
				# Track recent reward, action, and action log probability
				ep_rews.append(rew)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)


			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)
			
		batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		
		batch_rtgs = self.compute_rtgs(batch_rews)

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens
		
		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = False                             # If we should render during rollout
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results
		self.discrete = False							# Sets the type of environment to discrete or continuous
		self.training_step = 200						# Sets the number of trainig step

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.mean() for losses in self.logger['actor_losses']])
		actor_model = self.logger['actor_network']

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		#writer.add_scalar("Average Episodic Return", int(float(avg_ep_rews)), t_so_far)
		writer.add_scalar("Average actor Loss", int(float(avg_actor_loss)), t_so_far)
		# Tracking the weight of the network
		for name, param in actor_model.named_parameters():
			if 'weight' in name:
				writer.add_histogram(name, param.detach().cpu().numpy(), t_so_far)

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []

def test(env, actor_model, is_discrete):
	"""
		Tests the model.
		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in
		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	obs_dim = env.env.env.env.observation_space[0].shape[0]
	if is_discrete:
		act_dim = env.action_space.n
	else:
		act_dim = env.env.env.env.action_space[0].shape[0] #env.action_space.n #env.action_space.shape[0]

	# Build our policy the same way we build our actor model in PPO
	policy = FeedForwardActorNN(obs_dim, act_dim,is_discrete)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))
	

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	eval_policy(policy=policy, env=env, render=True, is_discrete=is_discrete)
		
