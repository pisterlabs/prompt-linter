#######version differer
import csv
import datetime
import numpy as np
import gym
import random
import MultiAgentEnv
import mujoco_py
import math
import tensorflow as tf
import tensorflow.contrib as tc
from actorpick import actor
from criticpick import critic
from collections import deque
env = gym.make('MultiAgentEnv-v0')
action_size = 4
state_size = 16
action_bound = env.action_space.high[:4]
print(action_bound)
batch_size = 128
import random
import matplotlib.pyplot as plt
###################seeding###################
seeding = 1234
np.random.seed(seeding)
tf.set_random_seed(seeding)
env.seed(seeding)

######################################

def cut_action_batch(batch):
    batch2 = np.empty([batch_size,action_size])
    for i in range(batch_size):
        batch2[i] = batch[i][:4]
    return batch2

#############This noise code is copied from openai baseline #########OrnsteinUhlenbeckActionNoise############# Openai Code#########

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

#########################################################################################################

def store_sample(s,a,r,d,info ,s2):
    x = s['observation'][3:]
    x2 = s2['observation'][3:]
    ob_1 = np.reshape(x,(1,16))
    ob_2 = np.reshape(x2,(1,16))
    replay_memory.append((ob_1,a,r,d,ob_2))

def get_pick_reward(s):
    if (s['achieved_goal'][2]) > 0.030:
       r = 0
    else:
       r = -1
    return r

def stg(s):
    x = s['observation'][3:]
    ob_1 = np.reshape(x,(1,16))
    return ob_1


def compute_dist(achieved_goal, goal):
      temp_desired = goal
      temp_achieved = achieved_goal
      eu_vector = []
      for i in range(len(temp_desired)):
            x = temp_desired[i]-temp_achieved[i]
            x = x*x
            eu_vector.append(x)
      sqr_sum = sum(eu_vector)
      sqrt = math.sqrt(sqr_sum)
      return sqrt


sess = tf.Session()
ac = actor(state_size, action_size, action_bound, sess,ini=True)
cr = critic(state_size, action_size, action_bound, sess,ini=True)
s = env.reset()

noice = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_size*2))

scores_path = 'model/single_ddpg/scores/pickscore-{date:%Y-%m-%d %H:%M:%S}.csv'.format( date=datetime.datetime.now() )

demo_actions = np.load('demoactions.npy')
demo_actions_counter = 0

demo_pick_actions = np.load('demopick.npy')
demo_pick_actions_counter = 0

demo_pick_flag=False

sess.run(tf.global_variables_initializer())
save_path = 'model/multi_ddpg/'
saver = tf.train.Saver()
replay_memory = deque(maxlen = 100000)
max_ep = 50000
max_ep_len = 250
ac.lam=0.1
ac.update_ac=0
demo_ep_threshold=-1
pick_ep_threshold=-1
gamma = 0.99
R_graph = deque(maxlen = 10)
R_graph_= []
saver.restore(sess, "model/multi_ddpg_pick/pick_model.ckpt")
for ii in range(max_ep):
    env = env.unwrapped
    env.set_random_goal(True)
    s = env.reset()
    R,r = 0,0
    for kk in range(max_ep_len):
        ss = stg(s)
        if ac.updateac>-1:
           a = ac.get_actions_(ss)
           b = a #+ noice()
           b[3] = a[3]
           b[7] = a[7]
           a = b
        else:
           a = ac.get_action(ss)
           a=a[0]
        env.render()
        s2,r,d,info=env.step(a)
    ac.updateac=0                              
    R_graph.append(R)
    R_graph_.append(R)
    ac.update_ac=0
    
