import numpy as np
import pickle as cPickle





class Powerplant:

    def __init__(self, args):
        self.args = args
        self.simulator = unpickle(args.simulator_filename)

    def reset(self):
        pass

    def step(self, action): #Expects a numpy action
        """Take an action to forward the simulation

            Parameters:
                control_input (ndarray): control action to take in the env

            Returns:
                next_obs (list): Next state
                reward (float): Reward for this step
                done (bool): Simulation done?
                info (None): Template from OpenAi gym (doesnt have anything)
        """
        next_states = self.simulator(action)


        pass


    def render(self):
        pass


def unpickle(filename):
    with open(filename, 'rb') as f:
        u = cPickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p
