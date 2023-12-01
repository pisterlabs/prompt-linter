"""
Code to interace with Box action and state spaces from OpenAI gym.
"""
import torch
import numpy as np

from pg_methods.interfaces import common_interfaces as common

class ContinuousProcessor(common.Interface):
    """
    Converts a gym Box observation into a pytorch representation
    """

    def gym2pytorch(self, gym_representation):
        """
        :param gym_representation: a numpy array of floats
        representing the state or action
        :return: the torch cast of the state
        """
        return torch.from_numpy(gym_representation).view(1, -1).float()

    def pytorch2gym(self, pytorch_representation):
        """
        Will return a gym numpy array
        representing the input pytorch_representation
        :param pytorch_representation:
        :return:
        """
        if isinstance(pytorch_representation, (np.ndarray, list)):
            # if len(pytorch_representation) == 1:
            #     # it is a single
            # if not len(pytorch_representation) == 1:
            #     raise IndexError('pytorch_representation is a list object, it must be of length 1. '
            #                      'Got the list: {}'.format(pytorch_representation) +
            #                      'Of type {}'.format(type(pytorch_representation)) +
            #                      'Otherwise use a custom processor')
            return np.array(pytorch_representation) #continous interfaces are numpy arrays
        else:
            return pytorch_representation.numpy().reshape(-1)

