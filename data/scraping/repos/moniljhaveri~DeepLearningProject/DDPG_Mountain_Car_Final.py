"""
The below script implements an agent that learns using the Deep Deterministic Policy Gradient algorithm.  The script
uses four classes: random noise generator, replay buffer, and actor and critic network classes.  After defining these
classes, the main loop runs the two-step learning cycle in which the agent (i) experiments with new actions and
evaluates them and then (ii) improves behavior based on the success of the experimentation.

The script is currently set to create a Mountain Car agent. 
"""

import tensorflow as tf
import numpy as np

# Import random noise generator class
from DDPG_Noise import OUNoise

# Import Replay Buffer class and deque data structure
import random
from memory import ReplayBuffer

# Import Actor and Critic network classes
from Actor import ActorNetwork
from Critic import CriticNetwork


#Import OpenAI gym
import gym
from gym import wrappers

import matplotlib.pyplot as plt
import math

        
# Learning Parameters

# Restore Variable used to load weights
RESTORE = False

# Number of episodes to be run
MAX_EPISODES = 700

# Max number of steps in each episode
MAX_EP_STEPS = 100000


# Learning rates
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001

# Size of replay buffer reflect how many transitions can be stored at once
BUFFER_SIZE = 10000
# Minibatch size is the number of transitions that are used to update the Q and policy functions
MINIBATCH_SIZE = 64

# Actor/Critical Neural Network Architecture
LAYER_1_SIZE = 400
LAYER_2_SIZE = 300

# Discount factor reflects the agents preference for short-term rewards over long-term rewards
GAMMA = 0.99

# Tau reflects how quickly target networks should be updated
TAU = 0.001

# Environment Variables

# Environment Name
# ENV_NAME = 'BipedalWalker-v2'
ENV_NAME = 'MountainCarContinuous-v0'
# Result storage locations
MONITOR_DIR = './results/biped_restart_8/gym_ddpg'
SUMMARY_DIR = './results/biped_restart_8/tf_ddpg'
RANDOM_SEED = 25


# The train function implements the two-step learning cycle.
def train(sess, env, actor, critic,RESTORE):
    
    sess.run(tf.global_variables_initializer())
    
    # Initialize random noise generator
    exploration_noise = OUNoise(env.action_space.shape[0])
    
    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay buffER
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    
    # Store q values for illustration purposes
    q_max_array = []
    
    for i in xrange(MAX_EPISODES):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        

        for j in xrange(MAX_EP_STEPS):

            env.render()

            # Begin "Experimentation and Evaluation Phase"
            
            # Seleect next experimental action by adding noise to action prescribed by policy 
            a = actor.predict(np.reshape(s, (1, actor.s_dim)))
            
            
            # If in a testing episode, do not add noise
            if i%100 is not 49 and i%100 is not 99:
                noise = exploration_noise.noise()
                a = a + noise


            # Take step with experimental action
            s2, r, terminal, info = env.step(np.reshape(a.T,newshape=(env.action_space.shape[0],)))

            # Add transition to replay buffer if not testing episode
            if i%100 is not 49 and i%100 is not 99:
                replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                                  terminal, np.reshape(s2, (actor.s_dim,)))

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if replay_buffer.size() > MINIBATCH_SIZE:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                    # Find target estimate to use for updating the Q-function
                    
                    # Predict_traget function determines Q-value of next state
                    target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                    # Complete target estimate (R(t+1) + Q(s(t+1),a(t+1)))
                    y_i = []
                    for k in xrange(MINIBATCH_SIZE):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + GAMMA * target_q[k])

                    # Perform gradient descent to update critic
                    predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))


                    ep_ave_max_q += np.amax(predicted_q_value, axis = 0)

                    # Perform "Learning" phase by moving policy parameters in direction of deterministic policy gradient
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)
                    actor.train(s_batch, grads[0])

                    # Update target networks
                    actor.update_target_network()
                    critic.update_target_network()

            s = s2
            ep_reward += r

            # If episode is finished, print results
            if terminal:
                
                if i%100 is 49 or i%100 is 99:
                    print("Testing")
                else:
                    print("Training")

                print '| Reward: %.2i' % int(ep_reward), " | Episode", i, '| Qmax: %.4f' % (ep_ave_max_q / float(j))
                q_max_array.append(ep_ave_max_q / float(j))
                
                break
    plt.plot(q_max_array)
    plt.xlabel('Episode Number')
    plt.ylabel('Max Q-Value')
    plt.show()
                
# Begin program                
def main():
    with tf.Session() as sess:

        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        # Check environment dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # print("Sample Action: ")
        # print(env.action_space.sample())
        # print("Sample Shape")
        # print(np.shape(env.action_space.sample()))
        # print("Valid Action")
        # val_act = np.array([[1.05],[0.5],[-1.3],[0.2]])
        # print(env.action_space.contains(val_act))
        # Ensure action bound is symmetric
        # assert (env.action_space.high == -env.action_space.low)

        
        # Build actor and critic networks
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
        
        

        
        # Film training videos if applicable
        env = wrappers.Monitor(env, MONITOR_DIR, force=True, video_callable=lambda episode_id: episode_id%49==0)
        

        train(sess, env, actor, critic,RESTORE)

            
main()


# In[ ]:




# In[ ]:



