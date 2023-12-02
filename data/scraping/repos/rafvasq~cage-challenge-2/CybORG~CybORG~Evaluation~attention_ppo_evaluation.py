import subprocess
import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG #, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.BlueLoadAgent import BlueLoadAgent
from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRemoveAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper

import ray
from ray import tune
import ray.rllib.algorithms.ppo as ppo
from ray.tune.registry import register_env
import numpy as np

MAX_EPS = 100
agent_name = 'Blue'

def env_creator(env_config):
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
    env = ChallengeWrapper(env=cyborg, agent_name='Blue')
    return env

register_env("cyborg", env_creator)

def wrap(env):
    return ChallengeWrapper(env=env, agent_name='Blue')

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

if __name__ == "__main__":
    cyborg_version = '2.0' #CYBORG_VERSION
    scenario = 'Scenario2'
    commit_hash = get_git_revision_hash()
    # ask for a name
    name = input('Name: ')
    # ask for a team
    team = input("Team: ")
    # ask for a name for the agent
    name_of_agent = input("Name of technique: ")

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Change this line to load your agent
    # agent = DQN.load("DQN against RedMeanderAgent")
    ray.init(num_cpus=3)
    config = {
        "env": "cyborg",
        "gamma": 0.99,
        "num_envs_per_worker": 20,
        "entropy_coeff": 0.001,
        "num_sgd_iter": 10,
        "vf_loss_coeff": 1e-5,
        "model": {
            # Attention net wrapping (for tf) can already use the native keras
            # model versions. For torch, this will have no effect.
            "_use_default_native_models": True,
            "use_attention": True,
            "max_seq_len": 10,
            "attention_num_transformer_units": 1,
            "attention_dim": 32,
            "attention_memory_inference": 10,
            "attention_memory_training": 10,
            "attention_num_heads": 1,
            "attention_head_dim": 32,
            "attention_position_wise_mlp_dim": 32,
        }
    }
    trained_model_path = "C:/Users/Rafael/ray_results/experiment_2022-06-20_12-54-22/experiment_cyborg_a23f4_00000_0_2022-06-20_12-54-22/checkpoint_001000/checkpoint-1000"
    agent = ppo.PPO(config=config, env="cyborg")
    agent.restore(trained_model_path)
    # start with all zeros as state
    num_transformers = config["model"]["attention_num_transformer_units"]
    attention_dim = config["model"]["attention_dim"]
    memory = config["model"]["attention_memory_inference"]
    init_state = state = [
        np.zeros([memory, attention_dim], np.float32)
        for _ in range(num_transformers)
    ]


    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)

            observation = wrapped_cyborg.reset()
            # observation = cyborg.reset().observation

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)
            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []
                # cyborg.env.env.tracker.render()
                for j in range(num_steps):
                    # action = agent.get_action(observation, action_space)
                    # action, _states = agent.predict(observation)
                    # action = agent.compute_single_action(observation) # RLlib 
                    action, state_out, _ = agent.compute_single_action(observation, state) # RLlib with Attention 
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    # result = cyborg.step(agent_name, action)
                    r.append(rew)
                    # r.append(result.reward)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()
            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')
