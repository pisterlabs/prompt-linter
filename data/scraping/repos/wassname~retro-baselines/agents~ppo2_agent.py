#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.

Run with:
```bash
python ppo2_agent.py   --config configs/sonic.json
```
"""

import tensorflow as tf

import os
import sys
import datetime as dt
import numpy as np
import logging

import baselines.ppo2_rudder.ppo2_rudder as ppo2_rudder
from baselines import bench, logger
from sonic_util import make_env as sonic_env

from TeLL.config import Config
from TeLL.utility.plotting import launch_plotting_daemon, save_subplots, save_movie, save_subplots_line_plots
from TeLL.utility.misc import make_sure_path_exists, Tee

# Start subprocess for plotting workers
#  Due to a garbage-collector bug with matplotlib/GPU, launch_plotting_daemon needs so be called before tensorflow
#  import
launch_plotting_daemon(num_workers=3)


# Log to file and stream
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger_file = logging.getLogger(__file__)


def train(env_id, num_timesteps, policy, working_dir, config):
    """Run PPO until the environment throws an exception."""
    # Original modules
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    import logging
    import gym
    import os.path as osp
    # Module modified for RUDDER
    from baselines.common.vec_env.vec_frame_stack import VecFrameStackNoZeroPadding
    from baselines.ppo2_rudder.policies import CnnPolicy, LstmPolicy, LstmPolicyDense

    bl_config = config.bl_config

    # Set numpy random seed
    rnd_seed = config.get_value('random_seed', 12345)
    rnd_gen = np.random.RandomState(seed=rnd_seed)

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get_value("cuda_gpu", "0"))

    # Tensorflow configuration
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=config.get_value("inter_op_parallelism_threads", 1),
        intra_op_parallelism_threads=config.get_value("intra_op_parallelism_threads", 1),
        log_device_placement=config.get_value("log_device_placement", False)
    )
    tf_config.gpu_options.allow_growth = config.get_value("tf_allow_growth", True)

    # Start Tensorflow session
    print("Preparing Logger...")

    gym.logger.setLevel(logging.INFO)
    print("Starting session...")
    tf_session = tf.Session(config=tf_config).__enter__()

    # Set tensorflow random seed
    tf.set_random_seed(rnd_seed)

    # Create parallel environments
    print("Preparing Envionments...", end="")

    def make_env(rank):
        def env_fn():
            np.random.seed(rnd_seed + rank)
            env = sonic_env(scale_rew=True)
            env.unwrapped.rank = rank
            env.seed(rnd_seed + rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            return env
        return env_fn

    nenvs = bl_config['num_actors']
    print("creating workers...", end="")
    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    set_global_seeds(rnd_seed)
    print("stacking frames...", end="")
    env = VecFrameStackNoZeroPadding(env, 4)
    print("Done!")

    # Enter learning
    policy = {'cnn': CnnPolicy, 'lstmdense': LstmPolicyDense, 'lstm': LstmPolicy}[policy]
    ppo2_rudder.learn(
        policy=policy,
        env=env,
        nsteps=1024,
        nminibatches=4,
        lam=0.95,
        gamma=0.99,
        noptepochs=4,
        log_interval=1,
        ent_coef=bl_config['ent_coef'],
        lr=lambda f: f * 6e-5 * bl_config['lr_coef'],
        cliprange=lambda f: f * 0.2,
        total_timesteps=int(num_timesteps * 1.1), tf_session=tf_session,
        working_dir=working_dir,
        config=config,
        plotting=dict(save_subplots=save_subplots, save_movie=save_movie,
                      save_subplots_line_plots=save_subplots_line_plots),
        rnd_gen=rnd_gen
    )


if __name__ == '__main__':
    config = Config()
    working_dir = os.path.join(config.working_dir, config.specs)
    working_dir = os.path.join(working_dir, dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    make_sure_path_exists(working_dir)

    with open(os.path.join(working_dir, 'log.txt'), 'a') as logfile:
        sys.stdout = Tee(sys.stdout, logfile, sys.stdout)

        bl_config = config.get_value('bl_config')

        logger.configure(os.path.join(working_dir, 'baselines'), ['tensorboard', 'log', 'stdout'])
        train(env_id=bl_config['env'], num_timesteps=bl_config['num_timesteps'],
              policy=config.get_value('policy'), working_dir=working_dir, config=config)

        sys.stdout.flush()
