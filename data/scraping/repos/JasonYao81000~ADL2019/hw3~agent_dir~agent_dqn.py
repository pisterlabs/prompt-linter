import random
random.seed(9487)
import math
import numpy as np
np.random.seed(9487)
from collections import namedtuple
import torch
torch.cuda.manual_seed_all(9487)
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from agent_dir.agent import Agent
from environment import Environment

use_cuda = torch.cuda.is_available()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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

class Dueling_DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(3136, 512)
        self.fc1_val = nn.Linear(3136, 512)

        self.fc2_adv = nn.Linear(512, num_actions)
        self.fc2_val = nn.Linear(512, 1)
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x

class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n
        self.double_dqn = args.double_dqn
        self.duel_dqn = args.duel_dqn
        if self.double_dqn and self.duel_dqn:
            self.model_name = 'double_duel_dqn'
        elif self.double_dqn:
            self.model_name = 'double_dqn'
        elif self.duel_dqn:
            self.model_name = 'duel_dqn'
        else:
            self.model_name = 'dqn'
        
        # Initialize your replay buffer
        self.memory_capacity = 10000
        self.memory = ReplayMemory(self.memory_capacity)

        # build target, online network
        self.target_net = Dueling_DQN(self.input_channels, self.num_actions) if self.duel_dqn else DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = Dueling_DQN(self.input_channels, self.num_actions) if self.duel_dqn else DQN(self.input_channels, self.num_actions)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load('./checkpoints/' + self.model_name)
        
        # discounted reward
        self.GAMMA = 0.99
        
        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = self.memory_capacity # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 300000

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
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            with torch.no_grad():
                action = self.online_net(state).max(1)[1].view(1, 1)
            return action[0, 0].data.item()

        # At first, you decide whether you want to explore the environemnt
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps / self.EPS_DECAY)

        # if explore, you randomly samples one action
        # else, use your model to predict action
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.online_net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)
            action = action.cuda() if use_cuda else action
        
        return action

    def update(self):
        # To update model, we sample some stored experiences as training examples.
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.uint8)
        non_final_mask = non_final_mask.cuda() if use_cuda else non_final_mask

        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to self.online_net
        state_action_values_ = self.online_net(state_batch)
        state_action_values = state_action_values_.gather(1, action_batch)

        with torch.no_grad():
            # Compute Q(s_{t+1}, a) for all next states.
            # Since we do not want to backprop through the expected action values,
            # use torch.no_grad() to stop the gradient from Q(s_{t+1}, a)
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.batch_size).cuda()
            if self.double_dqn:
                batch_index = torch.arange(self.batch_size, dtype=torch.long)[non_final_mask]
                selected_actions = torch.argmax(state_action_values_, dim=1)[non_final_mask]
                next_state_values[non_final_mask] = self.target_net(non_final_next_states)[batch_index, selected_actions].detach()
            else:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            
        # Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it is the terminal state.
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute temporal difference loss (Huber loss)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        episode_rewards = []
        loss = 0 
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            
            episode_reward = 0
            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action[0, 0].data.item())
                total_reward += reward
                episode_reward += reward

                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state
                if done:
                    next_state = None

                # store the transition in memory
                self.memory.push(state, action, next_state, torch.tensor([reward]).cuda())

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save('./checkpoints/' + self.model_name)

                self.steps += 1

            episode_rewards.append(episode_reward)

            if episodes_done_num % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))

                np.save('./results/' + self.model_name + '_episode_rewards.npy', np.array(episode_rewards))

                total_reward = 0

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        self.save('./checkpoints/' + self.model_name)
