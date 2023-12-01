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

# Creates enviroment from OpenAI gym
env = gym.make('CartPole-v0')

# Hyper parameters
learningRate = 0.80
discountRate = 0.99
exploreRate = 1.0
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
Q_value.add(Dense(48, input_dim = 4, activation='relu'))
Q_value.add(Dense(24, activation='relu'))
Q_value.add(Dense(2, activation='linear'))
Q_value.compile(loss='mse',  optimizer = Adam(lr = 0.001) )

# This is the memory replay buffer
memory = []

observation = env.reset()
observation = np.reshape(observation, [1, 4])
observationPrior = copy.deepcopy(observation)

# Either 0 or 1 - 0 is force left, 1 is force right
action = 0

# Generate training data and learn from it
while (gameNum < numEpisodes):
    
    env.render()

    if ( exploreRate > random.random() ):
        action = env.action_space.sample()
    else:
        # returns the index of the maximum element
        action = np.argmax( Q_value.predict( observation)[0] ) 

    observation, reward, done, info = env.step(action) 
    observation = np.reshape(observation, [1, 4])

    totalReward = totalReward + reward
    
    if ( done == True ):
        reward = -100
    
    # Add the training instance to our memory
    memory.append( trainingInstance(observationPrior, observation, reward, action, done, gameNum)  )
    
    # Sample from the memory and do an epoch of training
    batchSize = 20 # 20 #10
    batch = memory
    if ( len(memory) < batchSize):
        batch = memory
    else:
        batch = random.sample(memory, batchSize)

    for i in range(len(batch) ):
        value = Q_value.predict(batch[i].observationPrior)

        value[0][action] = batch[i].reward
    
        if ( (batch[i].action == 0) and (batch[i].done == False) ):
            value[0][0] = batch[i].reward + discountRate * np.amax(Q_value.predict(batch[i].observation)[0] )
        elif ( (batch[i].action == 1) and (batch[i].done == False) ):
            value[0][1] = batch[i].reward + discountRate * np.amax(Q_value.predict(batch[i].observation)[0] )
        
        Q_value.fit( batch[i].observationPrior, value, epochs = 2, verbose = 0)
    
    exploreRate = exploreRate * exploreDecayRate
    exploreRate = max(exploreRate, 0.01)

    observationPrior = copy.deepcopy(observation)

    if (done == True):
        observation = env.reset()
        observation = np.reshape(observation, [1, 4])
        observationPrior = copy.deepcopy(observation)
        
        gameNum = gameNum + 1
            
        # Format the string we will print
        sys.stdout.write("\033[F") # back to previous line
        sys.stdout.write("\033[K") # clear line        
        print("For the last completed episode: " + str(gameNum) +  " we scored " + str(totalReward) )
        allReward.append(totalReward)
        
        totalReward = 0
     
    # Periodically empty the memory buffer and up the explore rate
    if ( gameNum % 10 == 0):    
        memory = []
        exploreRate = 0.3


# Plot the data
plt.plot( allReward )
plt.ylabel('Cumulative Reward')
plt.xlabel('Episode')
plt.show()


env.close()
