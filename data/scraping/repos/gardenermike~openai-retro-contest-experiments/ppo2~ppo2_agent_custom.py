#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""

import tensorflow as tf

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import baselines.ppo2.ppo2 as ppo2
#import ppo2 as ppo2
import baselines.ppo2.policies as policies
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines import logger
import gym_remote.exceptions as gre
import math
from keras.layers import Conv2D, Dropout, Flatten, Dense

from sonic_util import make_env
#import retro
#from sonic_util_train import AllowBacktracking, SonicDiscretizer, RewardScaler, FrameStack, WarpFrame, make_env
import random
import csv
import sys


def get_training_envs():
    envs = []
    with open('./sonic-train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            game, state = row
            if game == 'game':
                continue
            else:
                envs.append(row)
    return envs

def make_training_env(game, state, stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = retro.make(game=game, state=state)
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)

    return env

def custom_cnn(obs_batch, **conv_kwargs):
    x = obs_batch

    # normalize
    x = (tf.cast(x, tf.float32) / 255.) - 0.5

    #x = tf.concat(x, axis=-1)

    initializer = tf.orthogonal_initializer(gain=math.sqrt(2))
    # activation=WaveNet_activation
    activation = 'relu'
    y = Conv2D(32, kernel_size=8, strides=4, activation=activation, kernel_initializer=initializer, name='layer_1')(x)
    y = Dropout(0.2)(y)
    y = Conv2D(64, kernel_size=4, strides=2, activation=activation, kernel_initializer=initializer, name='layer_2')(y)
    y = Dropout(0.1)(y)
    y = Conv2D(64, kernel_size=3, strides=1, activation=activation, kernel_initializer=initializer, name='layer_3')(y)
    y = Dropout(0.1)(y)

    y = Flatten(name='flatten')(y)
    y = Dense(512, activation='relu', kernel_initializer=initializer, name='dense1')(y)

    return y


class CustomCnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        self.pdtype = make_pdtype(ac_space)
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            #h = custom_cnn(X, **conv_kwargs)
            #print(conv_kwargs)
            h = policies.nature_cnn(X, **conv_kwargs)
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value

        #[<tf.Variable 'model/c1/w:0' shape=(8, 8, 4, 32) dtype=float32_ref>, <tf.Variable 'model/c1/b:0' shape=(1, 32, 1, 1) dtype=float32_ref>, <tf.Variable 'model/c2/w:0' shape=(4, 4, 32, 64) dtype=float32_ref>, <tf.Variable 'model/c2/b:0' shape=(1, 64, 1, 1) dtype=float32_ref>, <tf.Variable 'model/c3/w:0' shape=(3, 3, 64, 64) dtype=float32_ref>, <tf.Variable 'model/c3/b:0' shape=(1, 64, 1, 1) dtype=float32_ref>, <tf.Variable 'model/fc1/w:0' shape=(3136, 512) dtype=float32_ref>, <tf.Variable 'model/fc1/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'model/v/w:0' shape=(512, 1) dtype=float32_ref>, <tf.Variable 'model/v/b:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'model/pi/w:0' shape=(512, 7) dtype=float32_ref>, <tf.Variable 'model/pi/b:0' shape=(7,) dtype=float32_ref>]
        #<tf.Variable 'model/c1/w:0' shape=(8, 8, 1, 32) dtype=float32_ref>, <tf.Variable 'model/c1/b:0' shape=(1, 32, 1, 1) dtype=float32_ref>, <tf.Variable 'model/c2/w:0' shape=(4, 4, 32, 64) dtype=float32_ref>, <tf.Variable 'model/c2/b:0' shape=(1, 64, 1, 1) dtype=float32_ref>, <tf.Variable 'model/c3/w:0' shape=(3, 3, 64, 64) dtype=float32_ref>, <tf.Variable 'model/c3/b:0' shape=(1, 64, 1, 1) dtype=float32_ref>, <tf.Variable 'model/fc1/w:0' shape=(3136, 512) dtype=float32_ref>, <tf.Variable 'model/fc1/b:0' shape=(512,) dtype=float32_ref>

class MultigameEnvWrapper():
    def __init__(self):
        self.envs = get_training_envs()
        self.make_env()
        self.envs = get_training_envs()
        self.nsteps = 0
        self.switch_after_steps = 10000

    def make_env(self):
        game, state = random.choice(self.envs)
        self.env = make_training_env(game, state, stack=True, scale_rew=True)

    def step(self, *args):
        self.nsteps += 1
        if self.nsteps % self.switch_after_steps == 0:
            self.env.close()
            self.make_env()
            self.env.reset()
            
        return self.env.step(*args)


    def __getattr__(self, attr):
        return getattr(self.env, attr)

def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config):
        env = make_env
        #env = MultigameEnvWrapper
        #load_path = '/root/compo/trained_on_images_nature_cnn.joblib'
        load_path = './saved_weights_cnn.joblib'
        #load_path = './saved_weights.joblib'
        #logger.configure(dir='./logs', format_strs=['stdout', 'tensorboard'])

        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2.learn(policy=CustomCnnPolicy,
                   env=DummyVecEnv([env]),
                   nsteps=4096,
                   nminibatches=8,
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=3,
                   log_interval=1,
                   ent_coef=0.01, #0.2,
                   lr=lambda _: 2e-4,
                   cliprange=lambda i: 0.1, #1e-3,
                   total_timesteps=int(1e7),
                   load_path=load_path,
                   save_interval=500)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
