import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from agent_dir.agent import Agent
from environment import Environment

use_cuda = torch.cuda.is_available()
#torch.manual_seed(0)
#random.seed(0)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0
    
    def push(self, experience):
        # experience should be a list [st, at, r{t+1}, s{t+1}]
        # save the experience
        if len(self.memory) < self.capacity:
            # if still not full, just put the experience in
            self.memory.append(None)
        self.memory[self.pos] = experience
        self.pos = (self.pos + 1) % self.capacity
    
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

class AgentDDQN(Agent):
    def __init__(self, env, args, gamma=0.99, explore=[0.9,0.05,200], target_update_freq=1000, LR=1e-4, buffer_size=10000, plotting=False):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n

        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, self.num_actions)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load('dqn')

        # discounted reward
        self.GAMMA = gamma

        # EPS_START decay exponentially to EP_END for choosing action
        self.EPS_START = explore[0]
        self.EPS_END = explore[1]
        self.EPS_DECAY = explore[2]

        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 20000 # frequency to save the model
        self.target_update_freq = target_update_freq # frequency to update target network
        self.buffer_size = buffer_size # max size of replay buffer

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=LR)

        self.steps = 0 # num. of passed steps

        self.plotting = plotting

        # TODO: initialize your replay buffer
        self.replayBuffer = ReplayBuffer(self.buffer_size)


    def save(self, save_path):
        #print('save model to', save_path)
        #torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        #torch.save(self.target_net.state_dict(), save_path + '_target.cpt')
        return

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
        # TODO:
        # Implement epsilon-greedy to decide whether you want to randomly select
        # an action or not.
        # HINT: You may need to use and self.steps
        if test == True:
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) *\
             math.exp(-1. * self.steps / self.EPS_DECAY)
        #self.steps += 1
        if sample > eps_threshold:
            # action based on rule
            with torch.no_grad():
                values, indices = self.online_net(state.cuda()).max(1)
                action = indices.item()
                #print('action based on rule')
                #print(action)
        else:
            action = self.env.action_space.sample()
            #print('action based on random')
            #print(action)

        return action

    def update(self):
        # TODO:
        # step 1: Sample some stored experiences as training examples.
        experiences = self.replayBuffer.sample(self.batch_size)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        for experience in experiences:
            state_batch.append(experience[0])
            action_batch.append(experience[1])
            reward_batch.append(experience[2])
            next_state_batch.append(experience[3])
        """
        state_batch = [experience[0] for experience in experiences]
        action_batch = [experience[1] for experience in experiences]
        reward_batch = [experience[2] for experience in experiences]
        next_state_batch = [experience[3] for experience in experiences]
        """
        # step 2: Compute Q(s_t, a) with your model.
        state_batch_tensor = torch.cat(state_batch).cuda()
        value_online, indices_online = self.online_net(state_batch_tensor).max(1)
        Q_st = value_online
        # step 3: Compute Q(s_{t+1}, a) with target model.
        non_final_mask = torch.tensor(list(map(lambda s:s is not None, next_state_batch))).cuda()
        non_final_next_states = torch.cat([s for s in next_state_batch if s is not None]).cuda()
        
        next_state_values = torch.zeros(self.batch_size).cuda()
        #next_state_batch_tensor = torch.cat(next_state_batch).cuda()
        
        value_online, indices_online = self.online_net(non_final_next_states).max(1)
        value_target = self.target_net(non_final_next_states)
        
        value_target = value_target.gather(1, indices_online.view(-1,1))
        value_target = value_target.squeeze()
        # use indices to choose
        
        next_state_values[non_final_mask] = value_target
        #Q_st_1 = value_target
        
        # step 4: Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        reward_batch_tensor = torch.tensor(reward_batch).cuda()
        expected_Q = reward_batch_tensor + self.GAMMA * next_state_values
        # step 5: Compute temporal difference loss
        # HINT:
        # 1. You should not backprop to the target model in step 3 (Use torch.no_grad)
        # 2. You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it
        #    is the terminal state.
        
        #print(expected_Q)
        #print(Q_st)
        #THEloss = (expected_Q - Q_st).sum()
        loss_fn = torch.nn.MSELoss()
        #loss = F.smooth_l1_loss(Q_st, expected_Q)
        loss = loss_fn(Q_st, expected_Q)
        #print(loss)
        self.optimizer.zero_grad()
        self.target_net.eval()
        loss.backward()
        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.target_net.train()
        self.optimizer.step()
        #print(loss)

        return loss.item()

    def train(self):
        import matplotlib.pyplot as plt
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        total_reward_in_episode = 0
        window_size = 20 # size of window of moving average
        moving_reward = [] # compute moving average
        plot_list = [0] * window_size
        loss = 0
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)

            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                total_reward_in_episode += reward

                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)

                # TODO: store the transition in memory
                experience = [state, action, reward, next_state]
                self.replayBuffer.push(experience)

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and (self.steps % self.train_freq == 0):
                    loss = self.update()

                # TODO: update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    # target network should get the weight of policy network
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save('dqn'+str(self.steps))

                self.steps += 1

            if total_reward / self.display_freq > 30:
                self.save('dqn')
            
            if len(moving_reward) >= window_size:
                moving_reward.pop(0)
            moving_reward.append(total_reward_in_episode)
            total_reward_in_episode = 0
                
            if episodes_done_num % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))

                total_reward = 0

            # plot moving average reward
            if len(moving_reward) >= window_size:
                plot_list.append(sum(moving_reward)/len(moving_reward))
                """
                plt.plot(plot_list)
                plt.xlabel('number of episodes playing')
                plt.ylabel('average reward of last {} episodes'.format(window_size))
                plt.title('learning curve of dqn with pacman')
                plt.savefig('dqn-learning_curve.png')
                """
                yield plot_list

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break

        self.save('dqn')
