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
from collections import deque
env = gym.make('MultiAgentEnv-v0')
action_size = 4
state_size = 22
action_bound = env.action_space.high[:4]
print(action_bound)
batch_size = 128
import random
import matplotlib.pyplot as plt
from actorplace import actor
from criticplace import critic
from actorpick import actor as actorpick
from criticpick import critic as criticpick
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

def HER(environment, achieved, info):
    return environment.compute_reward(s['achieved_goal'], s['achieved_goal'], info)

def store_sample(s,a,r,d,info ,s2):
    ob_1 = np.reshape(s['observation'],(1,19))
    ac_1 = np.reshape(s['achieved_goal'],(1,3))
    de_1 = np.reshape(s['desired_goal'],(1,3))
    ob_2 = np.reshape(s2['observation'],(1,19))
    ac_2 = np.reshape(s2['achieved_goal'],(1,3))
    de_2 = np.reshape(s2['desired_goal'],(1,3))
    s_1 = np.concatenate([ob_1,ac_1],axis=1)
    s2_1 = np.concatenate([ob_2,ac_1],axis=1)
    s_2 = np.concatenate([ob_1,de_1],axis=1)
    s2_2 = np.concatenate([ob_2,de_1],axis=1)
    substitute_goal = s['achieved_goal'].copy()
    
    substitute_reward = HER(env, s['achieved_goal'],info)
   
    replay_memory.append((s_2,a,r,d,s2_2))
    replay_memory.append((s_1,a,substitute_reward,True,s2_1))

def stg(s):
    #print(len(s))
    ob_1 = np.reshape(s['observation'],(1,19))
    de_1 = np.reshape(s['desired_goal'],(1,3))
    return np.concatenate([ob_1,de_1],axis=1)


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

def hasPicked(s,i):
    if i >= 120:
       return True
    else:
       return False

save_path = 'model/multi_ddpg/'

sess = tf.Session()
sess2 = tf.Session()
ac = actor(state_size, action_size, action_bound, sess)
cr = critic(state_size, action_size, action_bound, sess)
saver = tf.train.Saver()
saver2 = tf.train.Saver()
ac_p = actorpick(state_size, action_size, action_bound, sess2,ini=True)
cr_p = criticpick(state_size, action_size, action_bound, sess2,ini=True)
s = env.reset()

noice = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_size))

ac_p.updateac=0
ac_p.lam=0.05
saver.restore(sess, "model/multi_ddpg/place_model_small.ckpt")
saver2.restore(sess2, "model/multi_ddpg/pick_model_small.ckpt")


replay_memory = deque(maxlen = 100000)
max_ep = 50000
max_ep_len = 1000
demo_ep_threshold=-1
pick_ep_threshold=-1
gamma = 0.99
R_graph = deque(maxlen = 10)
R_graph_= []


for ii in range(max_ep):
    picked = False
    env = env.unwrapped
    env.set_random_goal(True)
    env.set_random_object(False)
    s = env.reset()
    R,r = 0,0
    for kk in range(max_ep_len):
        #print('++')
        ss = stg(s)
       
   
        if not picked:
           a = ac_p.get_actions_(ss)
           picked = hasPicked(s,kk)
        else:
           a = ac.get_action(ss)
           #b = a + noice()
           b = a
           b[0][3] = a[0][3]
           b[0][7] = a[0][7]
           a = b
           a=a[0]
          
        #env.render()
        s2,r,d,info=env.step(a)
        r_2 = r
        r=r
        s = s2
        R += r_2
    ac_p.updateac=0
    print(ii, R, compute_dist(s['achieved_goal'],s['desired_goal']))
 
