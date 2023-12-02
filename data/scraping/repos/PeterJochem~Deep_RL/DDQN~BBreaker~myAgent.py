import numpy as np
import time
import random
import sys
import gym
import warnings
from PIL import Image
import cv2
from dataInstance import *
from observation import *
import copy
import signal, os
from replayBuffer import *

warnings.filterwarnings('ignore') # Gets rid of future warnings
with warnings.catch_warnings():
    from keras.layers import Activation, Dense, Flatten, Conv2D
    from keras.models import Sequential
    from keras.optimizers import RMSprop, Adam
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import matplotlib.pyplot as plt
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def handler1(signum, frame):

    print("The ctrl-z (sigstp) signal handler ran!")
    print("Here we should save the neural network and show what the agent has learned")
    # Add another boolean to switch back into training mode after pausing to view progress


signal.signal(signal.SIGTSTP, handler1)


class Agent():
    
    def __init__(self):
     
        # Hyper-parameters
        self.downSizedLength = 64
        self.downSizedWidth = 64
        self.maxNumGames = 1000000
    
        self.discount = 0.95
        self.memorySize = 500 # FIX ME
            
        self.epsilon = 1.0 # How often to explore        
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.1 # Don't let the epsilon be less than this
        self.batchSize = 499

        self.trainRate = 500 # After how many moves should we train? 
        self.moveNumber = 0 # Number of actions taken. Reset periodically 
        self.synchronize = 100 # The number of games to play before synchronizing networks
        self.save = 50 # This describes how often to save each network to disk

        self.replayMemory = replayBuffer(self.memorySize)
        
        self.currentGameNumber = 0

        self.cumulativeReward = 0 # Current game's total reward

        # These are the most recently seen observations 
        self.observationT1 = None
        self.observationT2 = None
        self.observationT3 = None
        self.observationT4 = None
        self.observationT5 = None
    
        # Creates enviroment from OpenAI gym
        self.env = gym.make('Breakout-v0')
        self.state = self.env.reset()

        # There are 4 actions for the Brickbreaker enviroment
        # Set this programatically
        self.action_space = 4

        # Define the two neural networks for Double Deep Q Learning
        self.QA = Sequential()
        self.QA.add(Conv2D(64, kernel_size = 3, activation = 'relu', input_shape = (64, 64, 4) ) )
        self.QA.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
        self.QA.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
        self.QA.add(Flatten())
        self.QA.add(Dense(self.action_space, activation = 'relu') )

        self.QA.compile(loss = "mean_squared_error",
        optimizer = RMSprop(lr = 0.00025,
        rho = 0.95,
        epsilon = 0.01),
        metrics = ["accuracy"])

        self.QB = copy.deepcopy(self.QA)

    
    def reorderObservations(self):
        """In order to make room for the next observation, move 
        back each one in the "queue" """

        self.observationT5 = self.observationT4
        self.observationT4 = self.observationT3
        self.observationT3 = self.observationT2
        self.observationT2 = self.observationT1
 
    # Generate the first three frames so we can perform an action
    # We need 4 frames to use the nueral network
    # Input: void
    # Return: void
    def generateFirst4Frames(self):
        
        self.env.render()
        action = self.env.action_space.sample()
        state, reward, done, info = self.env.step(action)
        self.observationT4 = observation(state)

        self.env.render()
        action = self.env.action_space.sample()
        state, reward, done, info = self.env.step(action)
        self.observationT3 = observation(state)
        
        self.env.render()
        self.action = self.env.action_space.sample()
        state, reward, done, info = self.env.step(action)
        self.observationT2 = observation(state)

        self.env.render()
        action = self.env.action_space.sample()
        state, reward, done, info = self.env.step(action)
        self.observationT1 = observation(state)

    def updateNetworks(self):
        QB = copy.deepcopy(self.QA)        
    
    
    def target_value(self, nextState, immediateReward):
        """ Use the Bellman equation to compute the optimal
        action and its predicted cumulative reward """
        
        # The network predicts the discounted value from that state
        # so predict the value for each state and then take the largest index with argmax
        action = np.argmax( self.QA.predict(np.array( [nextState] ) ) )
    
        return immediateReward + (self.discount * (self.QB.predict(np.array( [nextState] )))[0][action] )
        

    def train(self):
        """ Describe """
        
        # Get random sample from the ReplayMemory
        training_data = self.replayMemory.sample(self.batchSize)
        
        # Avoid training on the data garbage?
        # Add a check to make sure its not garbage!!!!!!!!

        # Split the data into input and label
        inputs = [""] * len(training_data)
        labels = [""] * len(training_data)

        for i in range(len(training_data) ):
            # faster way to do this?
            if (training_data[i] != None):
                if (training_data[i].data is not None):
                    # np.append copies the data!!!!!
                    #inputs = np.append(inputs, training_data[i].data.reshape((-1, 64, 64, 4)))
                    #print("adding a value")
                    #inputs.append( training_data[i].data.reshape(-1, 64, 64, 4))
                    #inputs.append( training_data[i].data )
                    #labels.append( training_data[i].value )
                    inputs[i] = training_data[i].data 
                    labels[i] = training_data[i].value

    
        inputs = np.array( inputs )
        labels = np.array( labels )
        # inputs = training_data[0].data.reshape((-1, 64, 64, 4))

        history = self.QA.fit(    
            inputs,
            labels,
            batch_size = 64,
            epochs = 5,
            verbose = 0
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            # validation_data=(x_val, y_val),
        )
             
    
        
    def saveNetworks(self):
        self.QA.save("neural_networks/A")
        self.QB.save("neural_networks/B")



myAgent = Agent()

# Add option to load network from saved files
# with command line arguments

# Generate the first four frames so we can perform an action
# We need 4 frames to use the neural network
myAgent.generateFirst4Frames()

myDataInstance = dataInstance(myAgent.observationT1, myAgent.observationT2, myAgent.observationT3, myAgent.observationT4)

while(True):
    myAgent.env.render()
    
    prediction = myAgent.QA.predict( np.array( [myDataInstance.data] ) )[0]
    action = np.argmax(prediction)
    
    if ( random.random() < myAgent.epsilon ):
        # Choose action randomly
        # FIX ME - depends on exact way the data is published in game
        action =  random.randint(0, myAgent.action_space - 1)
    
    
    state, reward, done, info = myAgent.env.step(action)
    
    # Label the data
    predictedValue = myAgent.target_value(myDataInstance.data, reward)
    
    myDataInstance.value = prediction
    myDataInstance.value[action] = predictedValue
    
    myAgent.replayMemory.append(myDataInstance)

    # Update counters
    myAgent.moveNumber = myAgent.moveNumber + 1
    myAgent.cumulativeReward = myAgent.cumulativeReward + reward
    
    # Prepare for the next iteration
    myAgent.reorderObservations()
    myAgent.observationT1 = observation(state) 
    
    # Turn most recent 4 observations into a training instance  
    myDataInstance = dataInstance(myAgent.observationT1, myAgent.observationT2, myAgent.observationT3, myAgent.observationT4)

    if (done):
        print("Game number " + str(myAgent.currentGameNumber) + " ended with a total reward of " + str(myAgent.cumulativeReward) )
        myAgent.env.reset()    
        myAgent.cumulativeReward = 0

        # Check if it is time to update/synchronize the two networks
        if ( (myAgent.currentGameNumber + myAgent.synchronize + 1) % myAgent.synchronize == 0 ):
            print(myAgent.currentGameNumber)
            print("Synchronized networks")
            # myAgent.updateNetworks()

        myAgent.currentGameNumber = myAgent.currentGameNumber + 1
         

    # Check if it is time to train
    if ( (myAgent.moveNumber + myAgent.trainRate) % myAgent.trainRate == 0 ):
        print("training")
        myAgent.epsilon = myAgent.epsilon * myAgent.epsilon_decay 
        if (myAgent.epsilon < myAgent.epsilon_min):
            myAgent.epsilon = myAgent.epsilon_min

        myAgent.train()

    # Check to save the network for later use
    if ( (myAgent.currentGameNumber + myAgent.save) % myAgent.save == 0 ):    
        pass
        # myAgent.saveNetworks()
    






