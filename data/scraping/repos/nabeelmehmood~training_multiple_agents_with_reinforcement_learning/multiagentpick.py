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
from actorpick import actor
from criticpick import critic
env = gym.make('MultiAgentEnv-v0')
action_size = 4
state_size = 16
action_bound = env.action_space.high[:4]
print(action_bound)
batch_size = 128
import random
import matplotlib.pyplot as plt
###################seeding################### Initializes random weights for neural networks
seeding = 1234
np.random.seed(seeding)
tf.set_random_seed(seeding)
env.seed(seeding)

############################################# Preprocessing for reshaping actions for training

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
#Stores the state/action pair into replay memory for training
def store_sample(s,a,r,d,info ,s2):
    x = s['observation'][3:]
    x2 = s2['observation'][3:]
    ob_1 = np.reshape(x,(1,16))
    ob_2 = np.reshape(x2,(1,16))
    replay_memory.append((ob_1,a,r,d,ob_2))

#Computes a virtual reward based on the y position of the object. If object is in air then successfully picked
def get_pick_reward(s):
    if (s['achieved_goal'][2]) > 0.028:
       r = 0
    else:
       r = -1
    return r

def stg(s):  #Reshaping observation vector from dict to numpy array
    x = s['observation'][3:]
    ob_1 = np.reshape(x,(1,16))
    return ob_1

 
def compute_dist(achieved_goal, goal):  #Used to compute virtual reward for HER
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

#Initialize tensorflow session
sess = tf.Session()
ac = actor(state_size, action_size, action_bound, sess,ini=True)
cr = critic(state_size, action_size, action_bound, sess,ini=True)
s = env.reset()
ac.updateac=0
noice = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_size*2))

scores_path = 'model/multi_ddpg_pick/scores/pickscore-{date:%Y-%m-%d %H:%M:%S}.csv'.format( date=datetime.datetime.now() ) #Path to save scores

#Load demos for training
demo_actions = np.load('demoactions.npy') 
demo_actions_counter = 0

demo_pick_actions = np.load('demopick.npy')
demo_pick_actions_counter = 0

demo_pick_flag=False 

sess.run(tf.global_variables_initializer()) #Initialize weights
save_path = 'model/multi_ddpg_pick/' #Model save path
saver = tf.train.Saver()
replay_memory = deque(maxlen = 100000)  #Number of items in replay memory - Requires a lot of RAM for higher numbers
max_ep = 50000 #Number of episodes to run
max_ep_len = 400 #Number of steps to perform in each episode
demo_ep_threshold=-1 #Chance to load demo ep
pick_ep_threshold=-1 #Chance to load picking demo ep
gamma = 0.99 #Discounted reward factor for estimating Q-Value
R_graph = deque(maxlen = 10) #Used for score printing purposes
R_graph_= []
#saver.restore(sess, "model/multi_ddpg_pick/model-2019-06-02 22:57:52.ckpt") #Can resume training from previous model
for ii in range(max_ep):
    env = env.unwrapped
    env.set_random_goal(True)  #Environment will initalize goal position randomly
    env.set_random_object(False) #Environment will initialize fixed position of object
    s = env.reset()
    demo_ep = False #Flags for demos
    demo_pick_ep = False
    cache_rand = random.randint(0,9) #Random number generator for demo
    if cache_rand <= demo_ep_threshold: 
        demo_ep = True
        demo_actions_counter = 0
    elif cache_rand <= pick_ep_threshold:
        demo_pick_ep = True
        demo_pick_actions_counter = 0

    R,r = 0,0
    for kk in range(max_ep_len):
        ss = stg(s)
        if ac.updateac>-1:
           a = ac.get_actions_(ss)
        else:
           a = ac.get_action(ss)
           b=a
           b[0][3] = a[0][3]           
           b = a #+ noice() #Used to add randomness to action 
           a = b
           a=a[0]
           demo_pick_flag=False
        #env.render() #Render the environment
        s2,r,d,info=env.step(a)
        
        if kk == 0 and r == 0:
           print("reset") #Reset if error in environment initialization
           s = env.reset()
           break
        
        r = get_pick_reward(s2)  #Get virtual reward
        r_2 = r
        store_sample(s,a,r,d,info,s2) #store state in replay memory
        s = s2
        R += r_2
          
        if batch_size < len(replay_memory): #Create batch of states for training
            minibatch = random.sample(replay_memory, batch_size)
            s_batch, a_batch,r_batch, d_batch, s2_batch = [], [], [], [], []
            for s_, a_, r_, d_, s2_ in minibatch:
               s_batch.append(s_)
               s2_batch.append(s2_)
               a_batch.append(a_)
               r_batch.append(r_)
               d_batch.append(d_)
            s_batch = np.squeeze(np.array(s_batch),axis=1)
            s2_batch = np.squeeze(np.array(s2_batch),axis=1)
            r_batch=np.reshape(np.array(r_batch),(len(r_batch),1))
            a_batch=np.array(a_batch)
            d_batch=np.reshape(np.array(d_batch)+0,(128,1))

    if s['achieved_goal'][2] > 0.038 and not demo_pick_ep:
            time_path = 'good-model-{date:%Y-%m-%d %H:%M:%S}.ckpt'.format( date=datetime.datetime.now() )
            saver.save(sess, save_path+ time_path)
            print('good model saved')

    #DDPG code to train
    a2 = ac.get_action_target(s2_batch) 
    v2 = cr.get_val_target(s2_batch,a2)
    tar= np.zeros((128,1))
    for o in range(128):
        tar[o] = r_batch[o] + gamma * v2[o]
    ac.updateac=0
    a_batch = cut_action_batch(a_batch)
    cr.train_critic(s_batch,a_batch,tar)
    a_out = ac.get_actions(s_batch)
    kk = cr.get_grad(s_batch,a_out)[0]
    ac.train_actor(s_batch, kk)
    cr.update_critic_target_net()
    ac.update_target_tar()                        
            
                                     
    R_graph.append(R) #Used for printing and calculating scores for output
    R_graph_.append(R)
    
    if ii % 10 ==0 : #Save model every 10 episodes
        time_path = 'model-{date:%Y-%m-%d %H:%M:%S}.ckpt'.format( date=datetime.datetime.now() )
        saver.save(sess, save_path+ time_path)
    print(ii, R, np.mean(np.array(R_graph)), np.max(np.array(R_graph)), s['achieved_goal'][2]) #Print scores
    if ii%1 == 0:
        with open(scores_path, mode='a+') as score_file: #Save scores
           score_writer = csv.writer(score_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
           score_writer.writerow([ii, np.mean(np.array(R_graph))])

    if (ii+1) % 100:
        plt.plot(np.array(R_graph_))
