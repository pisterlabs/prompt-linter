from ast import arg
import numpy as np
import os, sys
from create_env import create_env
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch
import wandb
from datetime import datetime
from pytz import timezone
from MakeTreeDir import MAKETREEDIR
from typing import Dict, Iterable, Optional, Union
import glob
import pickle

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""

def get_latest_run_id(log_path: Optional[str] = None, log_name: str = "") -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(f"{log_path}/{log_name}_[0-9]*"):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def get_env_params(env):
    env.reset()
    obs,_,_,info = env.step(env.action_space.sample())
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'g': obs['desired_goal'].shape[0],
            'obs_image': obs['image_observation'].shape,
            'action': env.action_space.shape[0],
            
            }
    #'action_max': env.action_space.high[0],
    for key, value in info.items():
        value= np.array(value)
        if value.ndim==0:
            value= value.reshape(1)
        params['info_{}'.format(key)] = value.shape[0]
    # params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    # create the ddpg_agent
    env = create_env()
    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    args.max_timesteps= env._max_episode_steps
    args.action_max = env.action_space.high[0]
    # create the ddpg agent to interact with the environment 
    wandb.init(project=f"Train_{args.project_name}" , name=f"{args.experiment_name}_index={index}_depth={args.depth}_depth_noise={args.depth_noise}_action_l2={args.action_l2}_{args.reward_type}_batch_size={args.batch_size}_lr={args.lr_actor}_texture_rand={args.texture_rand}_camera_rand={args.camera_rand}_light_rand={args.light_rand}_crop={args.crop_amount}", tags=["train",f"{args.env_name}"])
    wandb.config.update(args, allow_val_change=True)
    wandb.config.update(env_params,allow_val_change=True)
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn1()
    
    
    
if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    global ts
    global index
    now_asia = datetime.now(timezone(args.time_location))
    format = "%m-%d-%H:%M"
    ts = now_asia.strftime(format)
    global_dir = os.path.abspath(__file__ + args.global_file_loc )
    index = (get_latest_run_id(global_dir, args.experiment_name))
    if args.index is None:
        index = index+ 1
    else:
        index = args.index
    args.save_dir = os.path.join(global_dir,args.experiment_name+'_'+str(index))
    directory = MAKETREEDIR()
    directory.makedir(args.save_dir)
    
    print("------starting launch function")
    launch(args)
    
    
