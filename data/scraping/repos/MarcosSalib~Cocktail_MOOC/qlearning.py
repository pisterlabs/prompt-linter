#!/usr/bin/env python
# coding: utf-8

# ## Q-learning
# 
# This notebook will guide you through implementation of vanilla Q-learning algorithm.
# 
# You need to implement QLearningAgent (follow instructions for each method) and use it on a number of tests below.

# In[1]:


import sys, os
if 'google.colab' in sys.modules and not os.path.exists('.setup_complete'):
    get_ipython().system('wget -q https://raw.githubusercontent.com/yandexdataschool/Practical_RL/spring20/setup_colab.sh -O- | bash')

    get_ipython().system('wget -q https://raw.githubusercontent.com/yandexdataschool/Practical_RL/coursera/grading.py -O ../grading.py')
    get_ipython().system('wget -q https://raw.githubusercontent.com/yandexdataschool/Practical_RL/coursera/week3_model_free/submit.py')

    get_ipython().system('touch .setup_complete')

# This code creates a virtual display to draw game images on.
# It will have no effect if your machine has a monitor.
if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY")) == 0:
    get_ipython().system('bash ../xvfb start')
    os.environ['DISPLAY'] = ':1'


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


from collections import defaultdict
import random
import math
import numpy as np


class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly. 
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    #---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0
        
        value = max([self.get_qvalue(state, action) for action in possible_actions])

        return value

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        q_value = (1-learning_rate)*self.get_qvalue(state,action) + learning_rate * (reward + gamma * self.get_value(next_state))

        self.set_qvalue(state, action, q_value)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values). 
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        best_action_idx = np.argmax([self.get_qvalue(state, action) for action in possible_actions])
        best_action = possible_actions[best_action_idx]
        
        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.  
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list). 
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)
        action = None

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        if random.random() <= epsilon:
            chosen_action = np.random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state)
        
        return chosen_action


# ### Try it on taxi
# 
# Here we use the qlearning agent on taxi env from openai gym.
# You will need to insert a few agent functions here.

# In[4]:


import gym

try:
    env = gym.make('Taxi-v3')
except gym.error.DeprecatedEnv:
    # Taxi-v2 was replaced with Taxi-v3 in gym 0.15.0
    env = gym.make('Taxi-v2')

n_actions = env.action_space.n


# In[5]:


agent = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                       get_legal_actions=lambda s: range(n_actions))


# In[6]:


def play_and_train(env, agent, t_max=10**4):
    """
    This function should 
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        # get agent to pick action given state s.
        a = agent.get_action(s)

        next_s, r, done, _ = env.step(a)

        # train (update) agent for state s
        agent.update(s, a, r, next_s)
        
        s = next_s
        total_reward += r
        if done:
            break

    return total_reward


# In[7]:


from IPython.display import clear_output

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    agent.epsilon *= 0.99

    if i % 100 == 0:
        clear_output(True)
        print('eps =', agent.epsilon, 'mean reward =', np.mean(rewards[-10:]))
        plt.plot(rewards)
        plt.show()
        


# ### Submit to Coursera I: Preparation

# In[8]:


submit_rewards1 = rewards.copy()


# # Binarized state spaces
# 
# Use agent to train efficiently on CartPole-v0.
# This environment has a continuous set of possible states, so you will have to group them into bins somehow.
# 
# The simplest way is to use `round(x,n_digits)` (or numpy round) to round real number to a given amount of digits.
# 
# The tricky part is to get the n_digits right for each state to train effectively.
# 
# Note that you don't need to convert state to integers, but to __tuples__ of any kind of values.

# In[9]:


env = gym.make("CartPole-v0")
n_actions = env.action_space.n

print("first state:%s" % (env.reset()))
plt.imshow(env.render('rgb_array'))


# ### Play a few games
# 
# We need to estimate observation distributions. To do so, we'll play a few games and record all states.

# In[10]:


all_states = []
for _ in range(1000):
    all_states.append(env.reset())
    done = False
    while not done:
        s, r, done, _ = env.step(env.action_space.sample())
        all_states.append(s)
        if done:
            break

all_states = np.array(all_states)

for obs_i in range(env.observation_space.shape[0]):
    plt.hist(all_states[:, obs_i], bins=20)
    plt.show()


# ## Binarize environment

# In[11]:


from gym.core import ObservationWrapper


class Binarizer(ObservationWrapper):

    def observation(self, state):

        # state = <round state to some amount digits.>
        # hint: you can do that with round(x,n_digits)
        # you will need to pick a different n_digits for each dimension
        state[0] = np.round(state[0],0)
        state[1] = np.round(state[1],1)
        state[2] = np.round(state[2],2)
        state[3] = np.round(state[3],0)

        return tuple(state)


# In[12]:


env = Binarizer(gym.make("CartPole-v0"))


# In[14]:


all_states = []
for _ in range(1000):
    all_states.append(env.reset())
    done = False
    while not done:
        s, r, done, _ = env.step(env.action_space.sample())
        all_states.append(s)
        if done:
            break

all_states = np.array(all_states)

for obs_i in range(env.observation_space.shape[0]):

    plt.hist(all_states[:, obs_i], bins=20)
    plt.show()


# ## Learn binarized policy
# 
# Now let's train a policy that uses binarized state space.
# 
# __Tips:__ 
# * If your binarization is too coarse, your agent may fail to find optimal policy. In that case, change binarization. 
# * If your binarization is too fine-grained, your agent will take much longer than 1000 steps to converge. You can either increase number of iterations and decrease epsilon decay or change binarization.
# * Having 10^3 ~ 10^4 distinct states is recommended (`len(QLearningAgent._qvalues)`), but not required.
# * A reasonable agent should get to an average reward of >=50.

# In[19]:


agent = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                       get_legal_actions=lambda s: range(n_actions))


# In[20]:


# decay function for epsilon

eps_list = []
eps = 0.35
decay = 0.99999

for i in range(10000):
    eps = eps*decay**(i)
    eps_list.append(eps)

plt.plot(eps_list)


# In[21]:


rewards = []

for i in range(1000):
    
    # OPTIONAL YOUR CODE: adjust epsilon
    agent.epsilon = agent.epsilon*decay**(i)
    rewards.append(play_and_train(env, agent))
    
    if i % 100 == 0:
        clear_output(True)
        print('eps =', agent.epsilon, 'mean reward =', np.mean(rewards[-10:]))
        plt.plot(rewards)
        plt.show()
        


# ### Submit to Coursera II: Submission

# In[22]:


submit_rewards2 = rewards.copy()


# In[41]:


from submit import submit_qlearning
submit_qlearning(submit_rewards1, submit_rewards2, 'marcos.salib@outlook.com', '3pQNXC5R0DLmEf9W')


# In[ ]:




