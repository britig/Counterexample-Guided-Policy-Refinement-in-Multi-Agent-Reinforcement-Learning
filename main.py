import torch
from mappoPolicyTraining import MAPPO, test
from ppobaselines import PPOBaseline
from pettingzoo.sisl import multiwalker_v7
from pettingzoo.sisl import waterworld_v3
import supersuit as ss
from eval_policy import display
import pickle
from network import FeedForwardActorNN
import argparse
from UpdateNetwork import correct_policy
from Utility import compute_distance,combine_trajectories
import numpy as np
import yaml
#import optuna


if __name__ == "__main__":

	#=============================== Environment and Hyperparameter Configuration Start ================================#
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--env', dest='env', action='store_true', help='environment_name')
	parser.add_argument('--train', dest='train', action='store_true', help='train model')
	parser.add_argument('--train_baseline1', dest='train_baseline1', action='store_true', help='train baseline model')
	parser.add_argument('--train_baseline2', dest='train_baseline2', action='store_true', help='train baseline model')
	parser.add_argument('--test', dest='test', action='store_true', help='test model')
	parser.add_argument('--display', dest='display', action='store_true', help='Display Failure Trajectories')
	parser.add_argument('--subtrain', dest='subtrain', action='store_true', help='Training a subpolicy')
	parser.add_argument('--correct', dest='correct', action='store_true', help='Correct the orginal policy')
	parser.add_argument('--distance', dest='distance', action='store_true', help='Computing distance between two subpolicy')
	parser.add_argument('--combine', dest='combine', action='store_true', help='Combine Multiple trajectories into one trajectory')
	parser.add_argument('--type', dest='type', action='store_true', help='Type of Training')
	parser.add_argument('--secondary', dest='secondary', action='store_true', help='For secondary training')
	parser.add_argument('--actor', dest='actor', action='store_true', help='Actor Model')
	parser.add_argument('--critic', dest='critic', action='store_true', help='Critic Model')
	parser.add_argument('--oldactor', dest='oldactor', action='store_true', help='Old Actor Network')
	parser.add_argument('--oldcritic', dest='oldcritic', action='store_true', help='Old Critic Network')
	parser.add_argument('--subactor', dest='subactor', action='store_true', help='Sub Actor Network')
	parser.add_argument('--subcritic', dest='subcritic', action='store_true', help='Sub Critic Network')
	parser.add_argument('--newactor', dest='newactor', action='store_true', help='New Actor Network')
	parser.add_argument('--failuretraj', dest='failuretraj', action='store_true', help='File name with failure trajectories')
	args = parser.parse_args()
	if args.env:
		env_name = args.env
	else:
		env_name = 'multiwalker_v7'
	if args.actor:
		actor_model = args.actor
	else:
		actor_model = 'Policy/ppo_actormultiwalker_v72.pth'
	if args.critic:
		critic_model = args.critic
	else:
		critic_model = 'Policy/ppo_criticmultiwalker_v72.pth'
	if args.failuretraj:
		failure_trajectory = args.failuretraj
	else:
		failure_trajectory = 'Failure Trajectories Multiwalker/failure_trajectory_multiwalker_0_combined.data'
	if args.oldactor:
		old_actor = args.oldactor
	else:
		old_actor = 'Policy/ppo_actormultiwalker_v72.pth'
	if args.oldcritic:
		old_critic = args.oldcritic
	else:
		old_critic = 'Policy/ppo_criticmultiwalker_v72.pth'
	if args.subactor:
		sub_actor = args.subactor
	else:
		sub_actor = 'Policy/ppo_actor_subpolicymultiwalker_v7_primary.pth'
	if args.subcritic:
		sub_critic = args.subcritic
	else:
		sub_critic = 'Policy/ppo_critic_subpolicymultiwalker_v7_primary.pth'
	if args.newactor:
		new_actor = args.newactor
	else:
		new_actor = 'Policy/ppo_actor_subpolicymultiwalker_v7_safe.pth'
	if args.type:
		type = args.type
	else:
		type = 'secondary'
	with open('hyperparameters.yml') as file:
		paramdoc = yaml.full_load(file)
	
	# For multiwalker environment
	if env_name == 'multiwalker_v7':
		env = multiwalker_v7.env(n_walkers=2, position_noise=0, angle_noise=0,
		local_ratio=1.0, forward_reward=1.0, terminate_reward=-100.0, fall_reward=-30.0,
		terminate_on_fall=True, remove_on_fall=True, max_cycles=800)

	# For waterworld environment
	if env_name == 'waterworld_v3':
		env =  waterworld_v3.env(n_pursuers=2, n_evaders=4, n_poison=8, n_coop=2, n_sensors=4,
		sensor_range=0.2,radius=0.015, obstacle_radius=0.2,
		obstacle_coord=(0.5, 0.5), pursuer_max_accel=0.01, evader_speed=0.01,
		poison_speed=0.01, poison_reward=-1.0, food_reward=10.0, encounter_reward=0.01,
		thrust_penalty=-0.5, local_ratio=1.0, speed_features=True, max_cycles=500)

	is_discrete = False


	#=============================== Original Policy Training Code Start ================================#
	if args.train:
		for item, param in paramdoc.items():
			if(str(item)==env_name):
				hyperparameters = param
				print(param)
		model = MAPPO(False, '', env=env, **hyperparameters)
		model.learn(env_name, [])
	#=============================== Original Policy Training Code End ================================#
	#=============================== Baseline Policy Training Code Start ================================#

	# Baseline 1 reward+penalty
	if args.train_baseline1:
		for item, param in paramdoc.items():
			if(str(item)==env_name):
				hyperparameters = param
				print(param)
		model = PPOBaseline(False, '', env=env, **hyperparameters)
		model.learn(env_name, [])

	#Baseline 2 NeurIPS paper code
	if args.train_baseline2:
		env_sub_name = env_name+'-sub'
		for item, param in paramdoc.items():
			if(str(item)==env_sub_name):
				hyperparameters = param
				print(param)
		with open(failure_trajectory, 'rb') as filehandle1:
			# read env_state
			failure_observations = pickle.load(filehandle1)
		model = PPOBaseline( True, type, env=env, **hyperparameters)
		model.learn(env_name, failure_observations)
	#=============================== Baseline Policy Training Code End ================================#
	#=============================== Policy Testing Code Start ==========================#
	if args.test:
		test(env,actor_model, False)
	#=============================== Policy Testing Code End ==========================#

	#=============================== Displaying Failure Trajectories for Primary Objective Code Start  ==========================#
	if args.display:
		with open(failure_trajectory, 'rb') as filehandle1:
			# Read the failure trajectory
			failure_observations = pickle.load(filehandle1)
		print(f'Number of failure trajectories=========={len(failure_observations)}')
		obs_dim = env.env.env.env.observation_space[0].shape[0]
		act_dim = env.env.env.env.action_space[0].shape[0]
		policy = FeedForwardActorNN(obs_dim, act_dim,is_discrete)
		policy.load_state_dict(torch.load(actor_model))
		traj_count = 0
		count_fail_local = 0
		count_fail_mutual = 0
		count_fail_global = 0
		count_fail = 0
		traj_spec_dic = {}
		secondary_traj_dic = {}

		# For MultiWalker Environment
		if env_name == 'multiwalker_v7':
			for i in range(len(failure_observations)):
				seed = failure_observations[i][2]
				spec_type = failure_observations[i][4]
				print(seed)
				env.env.env.env.seed(seed[0])
				env.reset()
				# Extract the walker number and failure parameter
				for j in range(len(failure_observations[i][3])):
					print(failure_observations[i][3][j][0])
					env.env.env.env.walkers[failure_observations[i][3][j][1]].hull.angle = failure_observations[i][3][j][0]
				ep_ret, traj, iter  = display(failure_observations[i][0][0],policy,env,False,False)
				#Count the number of failure trajectories
				if ep_ret<0:
					secondary_traj_dic[count_fail] = failure_observations[i]
					if spec_type == 'local':
						print(f'Hull angle for walker 0 ==============={traj[-1][0]}')
						count_fail_local = count_fail_local+1
					if spec_type == 'mutual':
						print(f'Distance Between two walkers ==============={traj[-1][24]}======={traj[-1][26]}')
						count_fail_mutual = count_fail_mutual+1
					if spec_type == 'global':
						count_fail_global = count_fail_global+1
						print(f'Package angle ==============={traj[-1][30]}')
					count_fail = count_fail+1

		# For Waterworld environment
		if env_name == 'waterworld_v3':
			for i in range(len(failure_observations)):
				seed = failure_observations[i][2]
				spec_type = failure_observations[i][4]
				print(seed)
				env.env.env.env.seed(seed[0])
				env.reset()
				ep_ret, traj, iter  = display(failure_observations[i][0][0],policy,env,False,True)


		print(f'Number of Uncorrected Counterexmples ========== {count_fail}========{count_fail_local}====={count_fail_mutual}====={count_fail_global}')
		# Write the secondary failure trajecories in a file
		if type == 'primary':
			print('===========writing==========================')
			with open('failure_trajectory_multiwalker_walker0_secondary.data', 'wb') as filehandle2:
				# store the observation data as binary data stream
				pickle.dump(secondary_traj_dic, filehandle2)

	#=============================== Displaying Failure Trajectories Code End  ==========================#
	#=============================== Optuna Code (Not used) ========================================================#
	def objective(trial):
		hyperparameters = {
					'timesteps_per_batch': trial.suggest_int("timesteps_per_batch", 2048, 6144),
    				'max_timesteps_per_episode': 1002,
					'gamma': 0.99,
					'seed': 702,
					'training_step' : 500,
					"n_updates_per_iteration": 5,
					"lr": trial.suggest_float("lr", 0.0003, 0.0008),
					"clip": trial.suggest_float("clip", 0.1, 0.5),
		}
		with open(failure_trajectory, 'rb') as filehandle1:
			# read env_state
			failure_observations = pickle.load(filehandle1)
		model = MAPPO( True, type, env=env, **hyperparameters)
		return model.learn(env_name, failure_observations)
	#=============================== Optuna Code ========================================================#
	#=============================== Sub Policy Learning for Failure Trajectories Code Start  ==========================#
	if args.subtrain:
		type = "primary"
		env_sub_name = env_name+'-sub'

		for item, param in paramdoc.items():
			if(str(item)==env_sub_name):
				hyperparameters = param
		print(f'Hyperparameters ======== {hyperparameters}')
			
		with open(failure_trajectory, 'rb') as filehandle1:
			# read env_state
			failure_observations = pickle.load(filehandle1)
		model = MAPPO( True, type, env=env, **hyperparameters)
		model.learn(env_name, failure_observations)
		#study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(), direction="maximize")
		#study.optimize(objective, n_trials=20)
		#trial_ = study.best_trial
		#print(trial_.params)
	#=============================== Sub Policy Learning for Failure Trajectories Code End  ==========================#
	#=============================== Sub Policy Learning for Failure Trajectories Code Start  ==========================#
	if args.secondary:
		type = 'secondary'
		env_sub_name = env_name+'-sec'

		for item, param in paramdoc.items():
			if(str(item)==env_sub_name):
				hyperparameters = param
		print(f'Hyperparameters ======== {hyperparameters}')
			
		with open(failure_trajectory, 'rb') as filehandle1:
			# read env_state
			failure_observations = pickle.load(filehandle1)
		model = MAPPO( True, type, env=env, **hyperparameters)
		model.learn(env_name, failure_observations)
	#=============================== Sub Policy Learning for Failure Trajectories Code End  ==========================#
	#=============================== Policy Correction for Failure Trajectories Code Start  ==========================#
	if args.correct:
		correct_policy(env,old_actor,old_critic,sub_actor,sub_critic,is_discrete,failure_trajectory)
	#=============================== Policy Correction for Failure Trajectories Code End  ==========================#
	#=============================== Compute Distance between two policies Code Start  ==========================#
	if args.distance:
		#For finding out standard deviation
		distance_list = []
		for i in range(10):
			dist = compute_distance(old_actor,new_actor,env,is_discrete)
			distance_list.append(dist)
		distance_list =	np.array(distance_list)
		print(f'distance_list ========== {distance_list}')
		mean = np.mean(distance_list)
		std_dev = np.std(distance_list)
		print(f'mean dis ========== {mean} std div ======= {std_dev}')
	#=============================== Compute Distance between two policies Code End  ==========================#
	#=============================== Combine multiple failure trajetories into one file Code Start  ==========================#
	if args.combine:
		combine_trajectories('failure_trajectory_multiwalker_0_local.data','failure_trajectory_multiwalker_0_mutual.data','failure_trajectory_multiwalker_0_global.data')
	#=============================== Combine multiple failure trajetories into one file Code End  ==========================#

