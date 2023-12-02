import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import numpy as np
from random import sample

"""
Deep Q-Network Agent
[x] implements replay memory
[] actor-critic
[] distributed
"""
# helpers
def conv_size(in_size, padding, kernel_size, stride):
    return (in_size + 2*padding - kernel_size) / stride + 1

def pool_size(in_size, kernel_size, stride):
    return (in_size - kernel_size) / stride + 1

# !!NOTICE!!
# based on https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/learning_pytorch.py
# !!! -> by E. Culurciello, August 2017 <- !!!

class ReplayMemory:

    def __init__(self, capacity=10000, res=(40, 30), color_channels=3):
        shape = (capacity, color_channels, res[0], res[1])
        self.s1 = np.zeros(shape, dtype=np.float32)
        self.s2 = np.zeros(shape, dtype=np.float32)

        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.is_terminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, reward, s2, is_terminal):
        self.s1[self.pos, 0, :, :]     = s1
        if not is_terminal:
            self.s2[self.pos, 0, :, :] = s2

        self.a[self.pos]           = action
        self.is_terminal[self.pos] = is_terminal
        self.r[self.pos]           = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.r[i], self.s2[i], self.is_terminal


# base parameters
parameters = {
    'q': {'learning_rate': 0.00025,
        'discount_factor': 0.99,
        'epochs': 20,
        'learning_steps_per_epoch': 2000,
        'replay_memory_size': 10000
    },
    'nn': {'batch_size': 64}
}

def get_torch_var(ndarray):
    return Variable(torch.from_numpy(ndarray))


class DQN(nn.Module):

    def __init__(self, action_space_size, color_channels=3):
        super(DQN, self).__init__()
        self.action_space_size = action_space_size  # from OpenAI Gym
        self.color_channels = color_channels
        self.params = parameters
        self.epoch = 0

        # neural net configuration
        self.conv1 = nn.Conv2d(self.color_channels, 8, kernel_size=6, stride=3)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, action_space_size)

    @override
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 192)  # flattens out
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def _eval_dims(self, input_size):
        ex = Variable(torch.randn(1, 3, input_size, input_size))
        modules = [mod for mod in self.__dict__['_modules']][:-1]
        # excludes last linear module
        for module in modules:
            ex = self.__dict__['_modules'][module](ex)
        print('second-last layer is of shape', ex.size())
        dims = ex.size(1) * ex.size(2) * ex.size(3)
        print('the flattened layer will have', dims, 'dimensions')
        del(ex)

    def select_action(self):
        decayed_eps = self.hyperparams['eps_end'] + \
                      (self.hyperparams['eps_start'] - self.hyperparams['eps_end']) * \
                      np.exp(-1. * self.steps / self.hyperparams['eps_decay'])
        steps += 1
        if random.random() > decayed_eps:
            return self(Variable(state, volatile=True).type(FloatTensor))
                        .data.max(1)[1]
                        .view(1, 1)
        else:
            return LongTensor([[random.randrange(self.action_space_size)]])



class DQN_Agent(DQN):

    def __init__(self, action_space, color_channels = 3):
        super(DQN_Agent, self).__init__(action_space, color_channels)
        self.memory = ReplayMemory()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), learning_rate)


    def get_q_values(state):
        # state -> numpy array
        state = get_torch_var(state)
        return self(state)


    def get_best_action(state):
        q = get_q_values(state)
        m, index = torch.max(q, 1)
        action = index.data.numpy()[0]
        return action


    def learn(self, state, target_q):
        s1 = get_torch_var(state)
        target_q = get_torch_var(target_q)
        output = self(s1)
        loss = self.criterion(output, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss


    def learn_from_memory(self):
        """ Learns from a single transition. """
        # Get a random minibatch from the replay memory
        # and learn from it instead of the current game state
        if memory.size > batch_size:
            s1, a, r, s2, is_terminal = self.memory().get_sample()
            q = get_q_values(s2).data.numpy()
            q2 = np.max(q, axis=1)
            target_q = get_q_values(s1).data.numpy()
            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
            target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
            learn(s1, target_q)


    def get_exploration_rate(self):
        # eps standing for epsilon, exploration rate
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = .1 * epochs
        eps_decay_epochs = .6 * epochs

        if self.epoch < const_eps_epochs:
            return start_eps
        elif self.epoch < eps_decay_epochs:
            return start_eps - (epoch - const_eps_epochs) / \
                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        

    def perform_learning_step(self, game):
        s1 = game.get_screen()
        eps = self.get_exploration_rate()
        if random() <= eps:  # with probability eps
            a = randint(self.action_space_size)  # explore
        else:  # exploit
            s1 = s1.reshape([1, 1, game.down_res[0], game.down_res[1]])
            a = get_best_action(s1)

        r = game.make_action(actions[a])
        is_terminal = game.is_game_finished()
        s2 = game.get_screen()

        memory = memory.add_transition(s1, a, r, s2, is_terminal)
        self.learn_from_memory()


if __name__ == '__main__':

    dqn_agent = DQN_Agent(2)
    dqn_agent._eval_dims(100)