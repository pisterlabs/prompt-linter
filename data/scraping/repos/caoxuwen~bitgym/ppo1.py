#!/usr/bin/env python3

# Modified from OpenAI baseline ppo1/run_mujoco.py
# Need OpenAI baseline.
# Need mpi4py.
# IMPORTANT: Need flatten input of the policy network in
#     OpenAI baslines/ppo1/mlp_policy.py.
#     last_out = tf.contrib.layers.flatten(obz)
#   when the env observation length > 1.

from baselines.common import tf_util as U
from baselines import logger
import pandas as pd
import trading_env
import wrapper
import numpy as np

def train(training_env, num_timesteps):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    return pposgd_simple.learn(training_env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=1024,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=16, optim_stepsize=3e-4, optim_batchsize=512,
            gamma=1.0, lam=0.95, schedule='linear',
        )
    #training_env.close()

def evaluation(env, pi, num_times=1000):
    rewards = []
    for i in range(num_times):
      episode_reward = 0.0
      ob = env.reset()
      while True:
        action, _ = pi.act(True, ob)
        ob, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
          break
      print("evaluation ",i, ": ", episode_reward)
      rewards.append(episode_reward)
    print("evaluation: mu:", np.mean(rewards), "std:", np.std(rewards))
        
        

def main():
    df = pd.read_csv('dataset/btc_indexed2.csv')
    env = trading_env.make(env_id='training_v1', obs_data_len=50, step_len=1,
                           df=df, fee=0.003, max_position=5, deal_col_name='close',
                           return_transaction=True, sample_days=30, normalize_reward = True,
                           feature_names=['open', 'high', 'low', 'close', 'volume'])
    env = wrapper.LogPriceFilterWrapper(env)
    logger.configure()
    pi = train(env, num_timesteps=1024*1024)
    print("training done.")
    print(pi)


    df2 = pd.read_csv('dataset/btc_indexed2_test.csv')
    env2 = trading_env.make(env_id='training_v1', obs_data_len=50, step_len=1,
                           df=df, fee=0.003, max_position=5, deal_col_name='close',
                           return_transaction=True, sample_days=30, normalize_reward = False,
                           feature_names=['open', 'high', 'low', 'close', 'volume'])

    env2 = wrapper.LogPriceFilterWrapper(env2)
    evaluation(env2, pi, 50)

if __name__ == '__main__':
    main()
