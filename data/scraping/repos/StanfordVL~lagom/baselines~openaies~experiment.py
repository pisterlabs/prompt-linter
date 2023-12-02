import os
from pathlib import Path
from itertools import count
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import time

import gym
from gym.spaces import Box

import numpy as np
import torch
from lagom import Logger
from lagom.transform import describe
from lagom.utils import CloudpickleWrapper  # VERY IMPORTANT
from lagom.utils import pickle_dump
from lagom.utils import tensorify
from lagom.utils import set_global_seeds
from lagom.experiment import Config
from lagom.experiment import Grid
from lagom.experiment import Sample
from lagom.experiment import run_experiment
from lagom.envs import make_vec_env
from lagom.envs.wrappers import TimeLimit
from lagom.envs.wrappers import ClipAction
from lagom.envs.wrappers import VecMonitor
from lagom.envs.wrappers import VecStandardizeObservation

from baselines.openaies.openaies import OpenAIES
from baselines.openaies.agent import Agent


config = Config(
    {'log.freq': 10, 
     'checkpoint.num': 3,
     
     'env.id': Grid(['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'Swimmer-v3']), 
     'env.standardize_obs': False,
     
     'nn.sizes': [64, 64],
     
     # only for continuous control
     'env.clip_action': True,  # clip action within valid bound before step()
     'agent.std0': 0.6,  # initial std
     
     'train.generations': int(1e3),  # total number of ES generations
     'train.popsize': 64,
     'train.mu0': 0.0,
     'train.std0': 1.0,
     'train.lr': 1e-2,
     'train.lr_decay': 1.0,
     'train.min_lr': 1e-6,
     'train.sigma_scheduler_args': [1.0, 0.01, 450, 0],
     'train.antithetic': False,
     'train.rank_transform': True
     
    })


def make_env(config, seed):
    def _make_env():
        env = gym.make(config['env.id'])
        env = env.env  # strip out gym TimeLimit, TODO: remove until gym update it
        env = TimeLimit(env, env.spec.max_episode_steps)
        if config['env.clip_action'] and isinstance(env.action_space, Box):
            env = ClipAction(env)
        return env
    env = make_vec_env(_make_env, 1, seed)  # single environment
    return env
    
    
def initializer(config, seed, device):
    torch.set_num_threads(1)  # VERY IMPORTANT TO AVOID GETTING STUCK
    global env
    env = make_env(config, seed)
    env = VecMonitor(env)
    if config['env.standardize_obs']:
        env = VecStandardizeObservation(env, clip=5.)
    global agent
    agent = Agent(config, env, device)
    
    
def fitness(param):
    agent.from_vec(tensorify(param, 'cpu'))
    R = []
    H = []
    with torch.no_grad():
        for i in range(10):
            observation = env.reset()
            for t in range(env.spec.max_episode_steps):
                action = agent.choose_action(observation)['raw_action']
                observation, reward, done, info = env.step(action)
                if done[0]:
                    R.append(info[0]['episode']['return'])
                    H.append(info[0]['episode']['horizon'])
                    break
    return np.mean(R), np.mean(H)
    

def run(config, seed, device, logdir):
    set_global_seeds(seed)
    
    print('Initializing...')
    agent = Agent(config, make_env(config, seed), device)
    es = OpenAIES([config['train.mu0']]*agent.num_params, config['train.std0'], 
              {'popsize': config['train.popsize'], 
               'seed': seed, 
               'sigma_scheduler_args': config['train.sigma_scheduler_args'],
               'lr': config['train.lr'],
               'lr_decay': config['train.lr_decay'],
               'min_lr': config['train.min_lr'],
               'antithetic': config['train.antithetic'],
               'rank_transform': config['train.rank_transform']})
    train_logs = []
    checkpoint_count = 0
    with ProcessPoolExecutor(max_workers=config['train.popsize'], initializer=initializer, initargs=(config, seed, device)) as executor:
        print('Finish initialization. Training starts...')
        for generation in range(config['train.generations']):
            start_time = time.perf_counter()
            solutions = es.ask()
            out = list(executor.map(fitness, solutions, chunksize=4))
            Rs, Hs = zip(*out)
            es.tell(solutions, [-R for R in Rs])
            logger = Logger()
            logger('generation', generation+1)
            logger('num_seconds', round(time.perf_counter() - start_time, 1))
            logger('Returns', describe(Rs, axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('Horizons', describe(Hs, axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('fbest', es.result.fbest)
            train_logs.append(logger.logs)
            if generation == 0 or (generation+1)%config['log.freq'] == 0:
                logger.dump(keys=None, index=0, indent=0, border='-'*50)
            if (generation+1) >= int(config['train.generations']*(checkpoint_count/(config['checkpoint.num'] - 1))):
                agent.from_vec(tensorify(es.result.xbest, 'cpu'))
                agent.checkpoint(logdir, generation+1)
                checkpoint_count += 1
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    return None
    

if __name__ == '__main__':
    run_experiment(run=run, 
                   config=config, 
                   seeds=[1770966829, 1500925526, 2054191100], 
                   log_dir='logs/default',
                   max_workers=None,  # no parallelization 
                   chunksize=1, 
                   use_gpu=False,
                   gpu_ids=None)
