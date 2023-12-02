import wandb
import tensorflow as tf
import gym
import numpy as np

from pathlib import Path

from ppo_agent_raytest import Agent, writeout
from averaging import normal_avg, max_avg, softmax_avg, relu_avg
import ray
import argparse

import os
import sys

# quadcopter linking things
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
dir_name = os.path.join(parent_dir, '/Quadcopter_SimCon/Simulation/')

sys.path.append(parent_dir)
sys.path.append(parent_dir + '/Quadcopter_SimCon/Simulation/')


import time

# from pendulum_v1 import PendulumEquilEnv

# from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
# from gym_pybullet_drones.utils.Logger import Logger
# from gym_pybullet_drones.utils.utils import sync

# from gym_quad import GymQuad
# import Quadcopter_SimCon


tf.keras.backend.set_floatx('float64')

def main():
    try: wandb.finish()
    except: pass
    
    ####configurations

    group_temp = "022421-1_64-baselines12"
    # env_name = "Pendulum-v1"

    env_name = "Pendulum-v0"
    #     env_name = 'gym_quad-v0'
    # env_name = "CartPole-v2"

    wandb.init(group=group_temp, project="rl-ppo-federated", mode="online")
    

    wandb.config.gamma = 0.99
    wandb.config.update_interval = 16400 #16384
    wandb.config.actor_lr = 0.0003
    wandb.config.critic_lr = 0.0003

    wandb.config.batch_size = 64
    wandb.config.clip_ratio = 0.2
    wandb.config.lmbda = 0.95
    wandb.config.intervals = 10

    wandb.config.episodes = 82
    wandb.config.num = 1
    wandb.config.epochs = 70

    wandb.config.actor = {'layer1': 64, 'layer2' : 64}
    wandb.config.critic = {'layer1': 64, 'layer2' : 64, 'layer3': 16}
    
    wandb.config.average = "normal"    # normal, max, softmax, relu, epsilon
    wandb.config.kappa = 1      # range 1 (all avg) to 0 (no avg)
    wandb.config.epsilon = 0.2  # range from 1 to 0 (all random to never) - epsilon greedy

    wandb.run.name = wandb.run.id
    wandb.run.tags = [group_temp, "1-bot", "actor-64x2", "critic-64x2/16", "avg-normal", env_name]
    wandb.run.notes ="intense setup from openai, experiment w/ multiple parallel bots, more data exploited, trying the exact same critic"

    ISRAY = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--jobid', type=str, default=None)
    parser.add_argument('--actor_lr', type=float, default=None)
    parser.add_argument('--critic_lr', type=float, default=None)
    parser.add_argument('--clip_ratio', type=float, default=None)


    args = parser.parse_args()
    # print("args", args.jobid)

    if(args.jobid != None):
        wandb.config.jobid = args.jobid
        print("wandb jobid", wandb.config.jobid)
    if(args.actor_lr != None):
        wandb.config.actor_lr = args.actor_lr
        print("wandb actor_lr", wandb.config.actor_lr)
    if(args.critic_lr != None):
        wandb.config.critic_lr = args.critic_lr
        print("wandb critic_lr", wandb.config.critic_lr)
    if(args.clip_ratio != None):
        wandb.config.clip_ratio = args.clip_ratio
        print("wandb clip_ratio", wandb.config.clip_ratio)

    # print(wandb.config)
    if(ISRAY):
        ray.init(include_dashboard=False, ignore_reinit_error=True)
    # register_env("flythrugate-aviary-v0", lambda _: FlyThruGateAviary())
    
    # main run    
    N = wandb.config.num
    agents = []
    
    # gym.envs.register(
    #     id='Pendulum-v1',
    #     entry_point='pendulum_v1:PendulumEquilEnv',
    #     max_episode_steps=200
    # )   

    # gym.register(
    #     id="gym_quad-v0",
    #     entry_point = 'Quadcopter_SimCon.Simulation.gym_quad:GymQuad',
    # )

    # gym.envs.register(
    #     id='CartPole-v2',
    #     entry_point='continuous_cartpole:ContinuousCartPoleEnv',
    #     max_episode_steps=200
    # ) 

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    configuration = Struct(**wandb.config.as_dict())



    # set up the agent
    for i in range(N):
        # target = np.array([0, 0, 1, 0, 0, 0], dtype=np.float32)
        env_t = gym.make(env_name)
        # env_t.set_target(target)

        if(ISRAY):
            temp = Agent.remote(configuration, env_t, i)
            ref = temp.iden_get.remote()

            time.sleep(1)
            redo = True
            count = 0
            while(redo and count < 10):
                try:
                    ray.get(ref)
                    redo = False
                except:
                    print('mem error')
                    count += 1
                    time.sleep(1)
                    pass
            if(count >= 10):
                return 1
        else:
            temp = Agent(configuration, env_t, i)

        agents.append(temp)

    # early write out
    writeout(agents, 0)
    
    time.sleep(3)
    timesteps = 0
    # start the training
    max_reward = -np.inf
    for z in range(wandb.config.epochs):

        rewards = []
        
        buffer = {}
        buffer['states'] = []
        buffer['action'] = []
        buffer['reward'] = []
        buffer['old_policy'] = []
        buffer['gaes'] = []
        buffer['td_targets'] = []

        jobs = []
        # train the agent

        for j in range(wandb.config.episodes):
            jobs.append(agents[0].train.remote(max_episodes = 1))

        for i in range(wandb.config.episodes):
            rew, buff = ray.get(jobs[i])
            timesteps += 200
            print(timesteps)
            rewards.append(rew)

            buffer['states'] = buff['states']
            buffer['action'] = buff['action']
            buffer['reward'] = buff['reward']
            buffer['old_policy'] = buff['old_policy']
            buffer['gaes'] = buff['gaes']
            buffer['td_targets'] = buff['td_targets']

        # print(timesteps)
        # print(buffer)
        # print(rewards)

        for k in range(len(rewards)):
            wandb.log({'Reward' + str(j): rewards[k][0]})

        # if(ISRAY):
        #     for j in range(len(agents)):
        #         # print('Training Agent {}'.format(agents[j].iden))
        #         jobs.append(agents[j].train.remote(max_episodes = wandb.config.episodes))

        # for j in range(len(agents)):
        #     if(ISRAY):
        #         rewards.append(ray.get(jobs[j]))
        #     else:
        #         rewards.append(agents[j].train(max_episodes = wandb.config.episodes))

        #     for k in range(len(rewards[j])):
        #         wandb.log({'Reward' + str(j): rewards[j][k]})

        rewards = np.array(rewards, dtype=object)
        reward = np.average(rewards[:, -1])

        print('Epoch={}\t Average reward={}'.format(z, reward))
        wandb.log({'batch': z, 'Epoch-critic': reward})

        ray.get(agents[0].move_gradients.remote(buffer))

        # get the average - actor and critic
        if wandb.config.average == "max":
            critic_avg, actor_avg = max_avg(agents, rewards[:, -1])
        elif wandb.config.average == "softmax":
            print("softmax")
            critic_avg, actor_avg = softmax_avg(agents, rewards[:, -1])
        elif wandb.config.average == "relu":
            print("relu")
            critic_avg, actor_avg = relu_avg(agents, rewards[:, -1])
        elif wandb.config.average == "relu":
            print("relu")
            critic_avg, actor_avg = epsilon_avg(agents, rewards[:, -1], wandb.config.epsilon)
        else:
            critic_avg, actor_avg = normal_avg(agents)

        if z % 50 == 0:
            writeout(agents, z)
        
        jobs = []       
        # set the average
        if(ISRAY):
            for j in range(len(agents)):
                jobs.append(agents[j].actor_set_weights.remote(actor_avg, wandb.config.kappa))
                jobs.append(agents[j].critic_set_weights.remote(critic_avg, wandb.config.kappa))

            ray.wait(jobs, num_returns = 2 * len(agents), timeout=5000)
        else:
            for j in range(len(agents)):
                agents[j].actor_set_weights(actor_avg, wandb.config.kappa)
                agents[j].critic_set_weights(critic_avg, wandb.config.kappa)


        rewards = []
        jobs = []

        isrender = False
        if(ISRAY):
            for j in range(len(agents)):
                jobs.append(agents[j].evaluate.remote(render=isrender))


        for j in range(len(agents)):
            if(ISRAY):
                rewards.append(ray.get(jobs[j]))
            else:
                rewards.append(agents[j].evaluate(render=isrender))

        rewards = np.array(rewards, dtype=object)
        if(len(rewards) > 1):
            reward = np.average(rewards)
        else:
            reward = rewards

        print('Epoch={}\t Average reward={}'.format(z, reward))
        wandb.log({'batch': z, 'Epoch-avg': reward})

        if reward > max_reward:
            max_reward = reward
            writeout([agents[0]], z, "MAX")

        if z % 50 == 0:
            writeout([agents[0]], z, "average")
            
    writeout([agents[0]], wandb.config.epochs, "average")
    
    wandb.finish()


if __name__ == "__main__":
    
    main()
