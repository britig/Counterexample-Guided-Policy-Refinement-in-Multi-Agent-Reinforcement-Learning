"""
	Code for collecting failure trajectories using Bayesian Optimization
	Project : Policy correction using Bayesian Optimization
	Description : The file contains functions for computing failure trajectories given RL policy and
	safety specifications
"""

from GPyOpt.methods import BayesianOptimization
import numpy as np
from pettingzoo.sisl import waterworld_v3
from numpy.random import seed
import torch
from eval_policy import display
from network import FeedForwardActorNN
import pickle


'''
	Bayesian Optimization module for uncovering failure trajectories of Multiwalker environment

	Safety Requirement
	# The file handles three types of safety requirements
	# Local Requirement: Any walker should not fall down in any trajectory
	# Mutual Requirement: Two walkers should not crash along any trajectory
	# Global requirement: The package should not fall on the ground
'''

#=============================================Global Variables =================================#
policy = None
env = None
traj_spec_dic = {}
traj_count = 0
index_count = 0
number_of_walkers = 0
archea_bound = []
spec_type = ''


'''
	The function called from within the bayesian optimization module
	parameters : bounds containing the sampled variables of the state vector
	return : calls specification function and computes and returns the minimum value
'''
def sample_trajectory(bounds):
	global policy, env, traj_spec_dic,traj_count, index_count, archea_bound, spec_type
	selected_seed = env.env.env.env.seed(None)
	
	env.reset()

	#print(f'hull angle =========== {env.env.env.env.walkers[1].hull.angle}')

	# Set the bounds for specific walker
	# If there are multiple walkers specified in the walker bounds list
	archea_parameters = []	#stores a list containing tuples of parameters vs archea id's ex : [(0.5,0)]
	'''if len(archea_bound) > 1:
		for i in range(len(archea_bound)):
			env.env.env.env.walkers[archea_bound[i]].hull.angle = bounds[0][i]
			archea_parameters.append((bounds[0][i],archea_bound[i]))
	# If only one walker is specified in the walker bound list
	else:
		env.env.env.env.walkers[archea_bound[0]].hull.angle = bounds[0][0]
		archea_parameters.append((bounds[0][0],archea_bound[0]))'''
	
	obs, reward, d, info = env.last()
	#print(f'env.env.state =========== {env.env.state}')
	
	ep_ret, traj, iter = display(obs,policy,env,False,False)
	additional_data = {'reward':ep_ret}

	#Create trajectory to be sent to safety specification
	traj = (traj, additional_data)
	#print(f'trajectory ========== {traj}')

	# Trigger objective function as per specification type
	# local
	if spec_type == 'local':
		specification_evaluation = safet_spec_local(traj)
	# mutual
	elif spec_type == 'mutual':
		specification_evaluation = safet_spec_mutual(traj)
	# global
	else:
		specification_evaluation = safet_spec_global(traj)
	index_count = index_count+1
	#Store the set of trajectories with negative evaluation
	if specification_evaluation<0:
		traj_spec_dic[traj_count] = (traj[0],specification_evaluation,selected_seed,archea_parameters,spec_type)
		traj_count = traj_count + 1
	print(f'specification_evaluation ========== {specification_evaluation}')
	return specification_evaluation


# Function used for sampling disturbance parameters for the variables
def run_BO():
	bounds = [{'name': 'x1', 'type': 'continuous', 'domain': (0,1.4)}, #0.3,0.5 for global
	{'name': 'x2', 'type': 'continuous', 'domain': (0,1.4)}] # Hull angle
	max_iter = 200
	#max_time=100
	myProblem = BayesianOptimization(sample_trajectory, bounds, acquisition_type='EI', exact_feval=False, de_duplication = True)
	myProblem.run_optimization(max_iter, eps=1e-6, verbosity=True)
	myProblem.plot_convergence()
	print(myProblem.fx_opt)




# 1. Find the initial condition such that walker 0 falls
# That is the hull angle is greater the 0.7 for walker 0
def safet_spec_local(traj):
	traj = traj[0]
	hull_angle = traj[-1][0]
	return penalty

# 1. Find the initial condition such that the distance between the two walker is minimized
# Considering both left and right distances
# That is the distance between left neighbour should not be less than 0.23
def safet_spec_mutual(traj):
	traj = traj[0]
	penalty = (traj[-1][26] - 0.24)*10 #The crash value is 0.23 in most of the cases
	return penalty


# 1. Find the initial condition such that the package does not fall on ground
# Package angle not less than 0.25
def safet_spec_global(traj):
	reward = traj[1]['reward']
	traj = traj[0]
	penalty = reward - 10#Trajectories with less than 10 reward
	return penalty


# 1. Find the initial condition such that the reward is less than 50
def safet_spec_reward(traj):
	traj = traj[1]
	reward = traj['reward']
	#print(f'reward ========== {reward}')
	return -(50-reward)



if __name__ == '__main__':
	#Configuration parameters read from test_config.yml file 
	number_of_archea = 2
	# Which walker are we testing the disturbance for a list of walker for example [0,1] or single [0]
	archea_bound = [0]
	# Type of safety specification testing
	spec_type = 'global'
	
	actor_model = 'ppo_actorwaterworld_v3.pth'
	filename = 'failure_trajectory_waterworld_'+''.join(str(e) for e in archea_bound)+'_'+spec_type+'.data'
	print(filename)

	# Create environment
	env =  waterworld_v3.env(n_pursuers=2, n_evaders=4, n_poison=8, n_coop=2, n_sensors=4,
		sensor_range=0.2,radius=0.015, obstacle_radius=0.2,
		obstacle_coord=(0.5, 0.5), pursuer_max_accel=0.01, evader_speed=0.01,
		poison_speed=0.01, poison_reward=-1.0, food_reward=10.0, encounter_reward=0.01,
		thrust_penalty=-0.5, local_ratio=1.0, speed_features=True, max_cycles=500)

	# Extract out dimensions of observation and action spaces
	obs_dim = env.env.env.env.observation_space[0].shape[0]
	act_dim = env.env.env.env.action_space[0].shape[0]

	# Build our policy the same way we build our actor model in PPO
	policy = FeedForwardActorNN(obs_dim, act_dim, False)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))
	# Run the Bayesian Optimization function
	run_BO()
	print(f'Length trajectory ========== {len(traj_spec_dic)}')
	with open(filename, 'wb') as filehandle1:
		# store the observation data as binary data stream
		pickle.dump(traj_spec_dic, filehandle1)
