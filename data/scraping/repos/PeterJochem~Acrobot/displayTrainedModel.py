import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
import copy
import sys
import os
import logging
from trainingInstance import trainingInstance
import matplotlib.pyplot as plt
from keras.models import load_model

# Creates enviroment from OpenAI gym
env = gym.make('Acrobot-v1')

# Hyper parameters
learningRate = 0.80
discountRate = 0.99
exploreRate = 0.0
exploreDecayRate = 0.95

# This is the number of games to play before
# using the data to make graphs
numEpisodes = 200

# This is the total reward seen by the agent
totalReward = 0

# This will store the list of all the reward we 
# have seen over time
allReward = []

# This is the current game number
gameNum = 0

# Create the neural network
Q_value = Sequential()
Q_value.add(Dense(48, input_dim = 6, activation='relu'))
Q_value.add(Dense(24, activation='relu'))
Q_value.add(Dense(3, activation='linear'))
Q_value.compile(loss='mse',  optimizer = Adam(lr = 0.001) )

Q_value = load_model('models/myModel.h5')

observation = env.reset()
observation = np.reshape(observation, [1, 6])

# The action is either applying +1, 0 or -1 torque on the joint between
# the two pendulum links.
action = 0

# Generate training data and learn from it
while (gameNum < numEpisodes):
    
    env.render()

    # returns the index of the maximum element
    action = np.argmax( Q_value.predict( observation )[0] ) 

    observation, reward, done, info = env.step(action) 
    observation = np.reshape(observation, [1, 6])

    totalReward = totalReward + reward
    
    if (done == True):
        env.reset()
        observation = np.reshape(observation, [1, 6])
        
        gameNum = gameNum + 1
            
        # Format the string we will print
        sys.stdout.write("\033[F") # back to previous line
        sys.stdout.write("\033[K") # clear line        
        print("For the last completed episode: " + str(gameNum) +  " we scored " + str(totalReward) )
        allReward.append(totalReward)
        
        totalReward = 0
     
# Plot the data
#plt.plot( allReward )
#plt.ylabel('Cumulative Reward')
#plt.xlabel('Episode')
#plt.show()


env.close()
