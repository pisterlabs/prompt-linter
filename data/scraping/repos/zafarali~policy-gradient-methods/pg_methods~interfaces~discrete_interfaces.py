"""
Code to interace with discrete action and state spaces
from OpenAI gym.
"""
import torch
import numpy as np

from pg_methods.interfaces import common_interfaces as common

class SimpleDiscreteProcessor(common.Interface):
    def gym2pytorch(self, gym_representation):
        return torch.FloatTensor([common.number_convert(gym_representation)]).view(1, -1)

    def pytorch2gym(self, pytorch_representation):
        if isinstance(pytorch_representation, (np.ndarray, list)):
            if not len(pytorch_representation) == 1:
                raise IndexError('pytorch_representation is a list object, it must be of length 1. '
                                 'Otherwise use a custom processor')
            return pytorch_representation[0]
        else: #TODO: check if there is a less obvious way to do this than selecting one index
            return pytorch_representation.numpy()[0]

class OneHotProcessor(common.Interface):
    """
    Converts a gym number into the
    discrete one hot state embedding
    """
    def __init__(self, space_size, action_processor):
        """
        :param space_size: The size of the space
        :param action_processor: tells us if this is being used as an action processor
                                 The point of this is because when moving from observation to gym,
                                 we do move from the one hot to the int. However, when moving from action to gym
                                 the PyTorch multinomial policy returns ints that just need to be passed along rather
                                 than converted to a one hot.
        """
        self.space_size = space_size
        self.action_processor = action_processor

    def gym2pytorch(self, gym_representation, numpy=False):
        """
        :param gym_representation: an integer
        representing the state or action
        :param numpy: will return a numpy output rather than a torch one
        :return: the one hot encoding
        """
        vector_representation = np.zeros(self.space_size)
        vector_representation[gym_representation] = 1
        if not numpy:
            return torch.from_numpy(vector_representation).view(1, -1).float()
        else:
            return vector_representation.float()

    def pytorch2gym(self, pytorch_representation, numpy=False):
        """
        Will return a gym value representing the input vector
        :param pytorch_representation:
        :param numpy: will treat the input as a numpy array rather than a torch tensor
        :return:
        """
        if self.action_processor:
            try:
                # TODO(zafarali): Look for a way to make this less hacky.
                return pytorch_representation[0].item()
            except AttributeError:
                return pytorch_representation[0]
        if not numpy:
            return np.argmax(pytorch_representation.numpy())
        else:
            return np.argmax(pytorch_representation)


