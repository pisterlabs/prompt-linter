import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import copy
from agent_dir.agent import Agent
from environment import Environment
from logger import Logger

use_cuda = torch.cuda.is_available()


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

class Transition:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

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

        if args.test_dqn:
            self.load('dqnPDP2500')

        # discounted reward
        self.GAMMA = 0.99

        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.target_update_freq = 2500 # frequency to update target network
        self.buffer_size = 10000 # max size of replay buffer

        #loss 
        self.loss_function = nn.MSELoss()

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0 # num. of passed steps

        # TODO: initialize your replay buffer
        self.replay = [None] * self.buffer_size
        # prioritize 
        self.prior = [1] * self.buffer_size
        self.alpha = 0.6
        self.beta = 0.4
        self.max_prior = 1

        self.epsilon = 1

        #logger
        self.logger = Logger(f'dqnPDP{self.target_update_freq}')

    
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
        #action = self.env.action_space.sample()
        # TODO:
        # Implement epsilon-greedy to decide whether you want to randomly select
        # an action or not.
        # HINT: You may need to use and self.steps
        #print(test)
        state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
        state = state.cuda() if use_cuda else state
        probs = self.online_net(state)
        p = random.random()
        if not test and p < self.epsilon * (1-min(self.steps/(5*self.buffer_size),0.9)):
            action = random.randint(0,self.num_actions-1)
        else:
            prob, action = probs.topk(1)
            action = action[0].item()
            
            
        

        return action

    def update(self):
        # TODO:
        # step 1: Sample some stored experiences as training examples.
        # step 2: Compute Q(s_t, a) with your model.
        # step 3: Compute Q(s_{t+1}, a) with target model.
        # step 4: Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # step 5: Compute temporal difference loss
        # HINT:
        # 1. You should not backprop to the target model in step 3 (Use torch.no_grad)
        # 2. You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it
        #    is the terminal state.

        # prioritize
        prior = np.array(self.prior)
        prior = prior ** self.alpha
        prior = prior / np.sum(prior)
        #sampling_weight = (self.buffer_size*prior)**(-self.beta)
        #sampling_weight = sampling_weight/ np.sum(sampling_weight)
        # sample one batch
        tran_idx = random.choices(range(self.buffer_size),k=self.batch_size, weights=prior)
        #tran_idx = random.choices(range(self.buffer_size),k=self.batch_size)
        trans = [self.replay[idx] for idx in tran_idx]
        
        weights = [self.prior[idx] for idx in tran_idx]
        weights = (self.buffer_size * torch.FloatTensor(weights))**(-self.beta)
        weights = weights/ torch.max(weights)
        weights = weights.cuda() if use_cuda else weights
        #trans = random.choices(self.replay,k=self.batch_size)
        batch_state, batch_nextstate, rewards = torch.FloatTensor([]), torch.FloatTensor([]), torch.FloatTensor([])
        batch_idx = torch.LongTensor([])
        batch_state = batch_state.cuda() if use_cuda else batch_state
        batch_nextstate = batch_nextstate.cuda() if use_cuda else batch_nextstate
        for tran in trans:
            state = torch.from_numpy(tran.state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            batch_state = torch.cat((batch_state,state))

            batch_idx = torch.cat((batch_idx, torch.LongTensor([tran.action])))

            rewards = torch.cat((rewards, torch.FloatTensor([tran.reward])))

            next_state = torch.from_numpy(tran.next_state).permute(2,0,1).unsqueeze(0)
            next_state = next_state.cuda() if use_cuda else next_state
            batch_nextstate = torch.cat((batch_nextstate,next_state))

        dones = [tran.done for tran in trans]
        dones = torch.FloatTensor(dones).cuda() if use_cuda else torch.FloatTensor(dones)
        rewards = rewards.cuda() if use_cuda else rewards

        # current state 
        outputs = self.online_net(batch_state)
        Q = outputs[torch.arange(outputs.size(0)),batch_idx] 
        
        # next state
        # Double DQN
        next_a = self.online_net(batch_nextstate).detach()
        next_a = next_a.topk(1)[1].squeeze(1).long()
        outputs_hat = self.target_net(batch_nextstate).detach()
        Q_hat = outputs_hat[torch.arange(outputs_hat.size(0)),next_a]
        
        # Original
        """outputs_hat = self.target_net(batch_nextstate).detach()
        Q_hat = outputs_hat.topk(1)[0].squeeze(1)"""

        values = rewards + self.GAMMA * Q_hat* (1-dones)

        #loss = self.loss_function(values,Q )
        loss = torch.mean(weights*( values - Q )**2) 
        # update priorities
        diff = torch.abs(values- Q)
        for i,idx in enumerate(tran_idx):
            self.prior[idx] = diff[i].item()
        self.beta = min(self.beta + 0.01, 1)
        #print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        """for param in self.online_net.parameters():
            param.grad.data.clamp_(-1,1)"""
        self.optimizer.step()
        return loss.item()

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        loss = 0
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            
            done = False
            while(not done):
                
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                # clip reward, done in atari_wrapper.py (set True)
                reward = max(-1,min(reward,1))
                

                # process new state
                #next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                #next_state = state.cuda() if use_cuda else next_state
                # TODO: store the transition in memory
                tran = Transition(state, action, reward, next_state, done)
                self.replay[self.steps%self.buffer_size] = tran
                # For Prioritize Replay, new transition has maximal priority 
                self.prior[self.steps%self.buffer_size] = self.max_prior

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()
                    self.max_prior = max(self.prior)

                # TODO: update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save(f'dqnPDP{self.target_update_freq}')

                self.steps += 1

            if episodes_done_num % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
                self.logger.write('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f \n'%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))

                total_reward = 0

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        self.save(f'dqnPDP{self.target_update_freq}')

