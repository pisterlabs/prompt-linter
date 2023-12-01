import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from collections import namedtuple, deque

from agent_dir.agent import Agent
from environment import Environment

import pdb

use_cuda = torch.cuda.is_available()
# use_cuda = False # remove this if you can use gpu

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer():
    '''
    Fixed size buffer to store experience
    '''
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size) # use deque to store the newer experience

    def add(self, state, action, reward, next_state, done):
        e = Transition(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        return random.sample(self.memory, k=self.batch_size)


class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q

class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n

        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, self.num_actions)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        self.args = args
        if self.args.test_dqn:
            self.load('./ddqn') if use_cuda else self.load('./ddqn') 

        # discounted reward
        self.GAMMA = 0.99

        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network
        self.buffer_size = 10000 # max size of replay buffer

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

        self.steps = 0 # num. of passed steps

        self.epsilon = 0.95
        self.eps_end = 0.05
        self.eps_decay = 20000
        # TODO: initialize your replay buffer (experience replay)
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass

    def make_action(self, state, test=False):

        if test:
            state = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            with torch.no_grad():
                action = self.online_net(state).max(1)[1]
            return action.item()

        # state.shape = (1, 4, 84, 84)
        # output.shape = (1, 9)
        # TODO:
        # Implement epsilon-greedy to decide whether you want to randomly select
        # an action or not.
        # HINT: You may need to use and self.steps
        eps_threshold = np.random.uniform()
        eps_threshold = self.eps_end + (self.epsilon - self.eps_end) * math.exp(-1. * self.steps / self.eps_decay)
        
        if np.random.uniform() > eps_threshold: # choose action with largest prob
            with torch.no_grad():
                action = self.online_net(state).max(1)[1].view(1, 1).item()

            #     action = self.online_net(state)
            # action = torch.argmax(action).item()

        else: # choose action randomly
            action = np.random.randint(self.num_actions)

        return action

    def update(self):
        if len(self.memory.memory) < self.batch_size:
            return

        # TODO:
        # step 1: Sample some stored experiences as training examples.
        transition = self.memory.sample()
        batch = Transition(*zip(*transition))
        states = torch.cat(batch.state).cuda() if use_cuda else torch.cat(batch.state)
        actions = torch.Tensor(batch.action).cuda() if use_cuda else torch.Tensor(batch.action)
        rewards = torch.Tensor(batch.reward).cuda() if use_cuda else torch.Tensor(batch.reward)
        next_states = torch.cat(batch.next_state).cuda() if use_cuda else torch.cat(batch.next_state)
        dones = torch.Tensor(batch.done).cuda() if use_cuda else torch.Tensor(batch.done)
        

        # step 2: Compute Q(s_t, a) with your model.
        q_exp = self.online_net(states).gather(1, actions.long().view(-1, 1))
        
        with torch.no_grad():
            if self.args.ddqn:

                q_max_action = self.online_net(next_states).max(1)[1].unsqueeze(1)
                q_next = self.target_net(next_states).gather(1, q_max_action).reshape(-1)

            else:
                # step 3: Compute Q(s_{t+1}, a) with target model
                q_next = self.target_net(next_states).max(1)[0]
        # step 4: Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a)) (discounted)
        q_target = rewards + (self.GAMMA * q_next * (1-dones))
        # step 5: Compute temporal difference loss
        loss = self.criterion(q_exp, q_target.unsqueeze(1))
        # HINT:
        # 1. You should not backprop to the target model in step 3 (Use torch.no_grad)
        # 2. You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it
        #    is the terminal state.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        print('ddqn method: {}'.format(self.args.ddqn))
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        episode_rewards = []
        loss = 0
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state

            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state

                # TODO: store the transition in memory
                self.memory.add(state, action, reward, next_state, done)

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # TODO: update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save('./ddqn')

                self.steps += 1

            episode_rewards.append(total_reward)

            if episodes_done_num % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
                total_reward = 0
                np.save('./reward/ddqn-reward.npy', np.array(episode_rewards))

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        self.save('./ddqn')
