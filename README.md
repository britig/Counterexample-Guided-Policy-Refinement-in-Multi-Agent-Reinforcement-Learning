# Counterexample-Guided-Policy-Refinement-in-Multi-Agent-Reinforcement-Learning
Code contributions and supplement material for counterexample-guided policy refinement in multi-agent reinforcement learning (Accepted in AAMAS 2023)

# System Requirements

The code has been tested in systems with the following OS

- Ubuntu 20.04.2 LTS
- Windows 10 with Anaconda

## Installation

1. Setup conda environment

```
$ conda create -n env_name python=3.8.5
$ conda activate env_name
```
2. Clone the repository to an appropriate folder
3. Navigate to the appropriate environment folder like multiwalker and Install requirements

```
$ pip install -r requirements.txt
$ pip install -e .
```

4. All code should be run from the respective environment folder. The output files (policies and failure trajectory files are also saved inside this folder).

## Usage

All the trained policies, sub-policies and updated policies are avialable in the Policy folder

The failure trajectories are present in Failure Trajectories Multiwalker folder

The main program takes the following command line arguments

1) --env : environment name (default is multiwalker_v7)
2) --actor : filepath to the actor network (default is Policy/ppo_actormultiwalker_v72.pth)
3) --critic : filepath to the critic network (default is Policy/ppo_criticmultiwalker_v72.pth)
4) --failuretraj : The filepath to the failure trajectory file (default is Failure Trajectories Multiwalker/failure_trajectory_multiwalker_0_combined.data)
5) --isdiscrete : True if environment is discrete (default False)
6) --oldactor : filepath to the original actor network (default is Policy/ppo_actormultiwalker_v72.pth)
7) --oldcritic : filepath to the original critic network (default is Policy/ppo_criticmultiwalker_v72.pth)
8) --subactor : filepath to the subpolicy actor network (default is Policy/ppo_actor_subpolicymultiwalker_v7_primary.pth)
9) --subcritic : filepath to the subpolicy critic network (default is Policy/ppo_critic_subpolicymultiwalker_v7_primary.pth)
10) --newactor : filepath to the updated actor network (default is Policy/ppo_actor_policymultiwalker_v7_safe.pth)

The hyperparameters can be changed in the hyperparameters.yml file


Note : Change the default arguments inside the main.py file otherwise the command line may become too long



### Training the sub-policy

Set --type argument to primary for primary training
Set --failuretraj : The filepath to the failure trajectory path (default is Failure Trajectories Multiwalker/failure_trajectory_multiwalker_0_combined.data)

```
$ python main.py --subtrain
```



### Update the original Policy to correct type 1 trajectories

```
$ python main.py --correct
```
The correct method takes the actor and critic networks of the old policy and the subpolicy as an argument

default function parameters are 
1) --env : environment name (default is multiwalker_v7)
2) --oldactor : filepath to the original actor network (default is Policy/ppo_actormultiwalker_v72.pth)
3) --oldcritic : filepath to the original critic network (default is Policy/ppo_criticmultiwalker_v72.pth)
4) --subactor : filepath to the subpolicy actor network (default is Policy/ppo_actor_subpolicymultiwalker_v7_primary.pth)
5) --subcritic : filepath to the subpolicy critic network (default is Policy/ppo_critic_subpolicymultiwalker_v7_primary.pth)
6) --failuretraj : The filepath to the failure trajectory path (default is Failure Trajectories Multiwalker/failure_trajectory_multiwalker_0_combined.data)
7) --isdiscrete : True if environment is discrete (default False)
8) --type : Primary during type 1 correction

### Update the original Policy to correct type 2 trajectories

First run the --display argument with type primary and updated actor model ex: --actor = 'Policy/ppo_actor_updatedmultiwalker_0_combined_v7.pth' to generate type 2 trajectories

```
$ python main.py --secondary
```

Set the following variables 

1) --type = secondary
2) --failuretraj : The filepath to the secondary failure trajectory path (ex : Failure Trajectories Multiwalker/failure_trajectory_multiwalker_walker0_secondary.data)

### Calculate the distance between the original policy and the updated policy

```
$ python main.py --distance
```
default function parameters are:
1) --oldactor : filepath to the original actor network (default is Policy/ppo_actormultiwalker_v72.pth)
2) --newactor : filepath to the updated actor network (default is Policy/ppo_actor_subpolicymultiwalker_v7_safe.pth)
3) --env : environment name (default is multiwalker_v7)
4) --isdiscrete : True if environment is discrete (default False)

### Generating Failure trajectories for a specific environment and policy

Failure trajectories uncovered with our tests are available in Failure_Trajectories Folder

Each environment has a seperate Bayesian Optimization file. Run the Bayesian Optimization correspondig to the environment
We use GpyOpt Library for Bayesian Optimization. As per (https://github.com/SheffieldML/GPyOpt/issues/337) GpyOpt has stochastic evaluations even when the seed is fixed.
This may lead to identification of a different number failure trajectories (higher or lower) than the mean number of trajectories reported in the paper.

For example to generate failure trajectories for the Multiwalker environment run:

```
$ python MultiwalkerTesting.py --spectype=local
```

spectype argument can be set to local, global or mutual. default is "local".
set --actor : filepath to the actor network to be tested (default is Policy/ppo_actormultiwalker_v72.pth)

The failure trajectories will be written in the corresponding data files in the same folder. 
To combine the failure trajectories run 

```
$ python main.py --combine
```
 
with local, mutual and global trajectories file names as function parameters

### Displaying Failure trajectories

To display failure trajectories:

```
$ python main.py --display
```
Mention the actor policy and the failure trajectory file in arguments or in the main.py file

Change the actor_model argument for observing the behaviour of sub-policy, updated policy and the safe policy on the failure trajectories

### Testing

et --actor : filepath to the actor network to be tested (default is Policy/ppo_actormultiwalker_v72.pth)
To test a trained model run:

```
$ python main.py --test
```

Press ctr+c to end testing



### Training a policy from scratch

To train a model run:

```
$ python main.py --train
```
The hyperparameters can be changed in the hyperparameters.yml file
