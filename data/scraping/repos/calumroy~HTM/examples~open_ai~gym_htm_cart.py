import sys
import json
from HTM_network import HTM_network
from utilities import startHtmGui as htmgui
from utilities import openAiSimulator as openAiSim
import math
import gym
import os


'''
This python script creates and runs the open ai cart simulation with
the htm network.

'''


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def main():
    # Define the open ai gym simulation type
    env = gym.make('MountainCar-v0')

    # define the gui class
    gui = htmgui.start_htm_gui()

    # Open and import the parameters .json file
    param_file = find('gym_htm_cart.json', './')
    #import ipdb; ipdb.set_trace()
    with open(param_file, 'r') as paramsFile:
        params = json.load(paramsFile)

    # Define the size of the input grids to the htm. This
    # is the size of the boolean matrix that the simualtion values are encoded into.
    inputWidth = 60
    inputHeight = 15
    # Define the cart simulation run times.
    num_episodes = 2
    max_time_per_epi = 500

    # Define the Input creator.
    random_actions = True
    InputCreator = openAiSim.openAiSimulator(env,
                                             num_episodes,
                                             max_time_per_epi,
                                             inputWidth, inputHeight, random_actions)

    # Create a HTM object
    htm = HTM_network.HTM(InputCreator.encoder.encodeVar(0), params)

    # print(env.action_space)
    # print(env.observation_space)

    # Start the htm gui and open ai simulation
    gui.startHtmGui(htm, InputCreator)

if __name__ == '__main__':
    main()


