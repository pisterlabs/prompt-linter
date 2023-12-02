# checkout https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/tree/master/DDQN
# I didn't make many changes, I simply ensured it fits with the CybORG BaseAgent

import inspect

from CybORG import CybORG
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.Wrappers.ChallengeWrapper import ChallengeWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
import torch as T
import numpy as np
from DQN.DeepQNetwork import DeepQNetwork, DeepRNNNetwork
from DQN.ReplayBuffer import ReplayBuffer

class DQNAgent(BaseAgent):
    def __init__(self, gamma=0.9, epsilon=0, lr=0.1, n_actions=41, input_dims=(52,),
                 mem_size=1000, batch_size=32, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo='DDQN', env_name='Scenario1b', chkpt_dir='chkpt', load=False):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                        input_dims=self.input_dims,
                                        name=self.env_name+'_'+self.algo+'_q_eval',
                                        chkpt_dir=self.chkpt_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                        input_dims=self.input_dims,
                                        name=self.env_name+'_'+self.algo+'_q_next',
                                        chkpt_dir=self.chkpt_dir)

    # if epsilon=0 it will just use the model
    def get_action(self, observation, action_space, ignore_epsilon=False):
        if ignore_epsilon or (np.random.random() > self.epsilon):
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def train(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)
        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()


class RNNDQNAgent(DQNAgent):
    def __init__(self, gamma=0.99, epsilon=1, lr=0.0001, n_actions=10, input_dims=(10), lookback_steps=7,
                 mem_size=1000, batch_size=64, eps_min=0.01, eps_dec=5e-7, hid_size=64,
                 replace=1000, algo=None, env_name=None, chkpt_dir='chkpt', load=False,
                 env=None):
        self.lookback_steps = lookback_steps

        super(RNNDQNAgent, self).__init__(gamma=gamma, epsilon=epsilon, lr=lr, n_actions=n_actions,
                input_dims=input_dims,
                 mem_size=mem_size, batch_size=batch_size, eps_min=eps_min, eps_dec=eps_dec,
                 replace=replace, algo=algo, env_name=env_name, chkpt_dir=chkpt_dir)

        self.memory = ReplayBuffer(mem_size, (self.lookback_steps, input_dims[0]), n_actions)

        self.q_eval = DeepRNNNetwork(self.lr, self.n_actions,
                                        input_dims=self.input_dims,
                                        name=self.env_name+'_'+self.algo+'_q_eval',
                                        chkpt_dir=self.chkpt_dir, hid_size=hid_size)
        self.q_next = DeepRNNNetwork(self.lr, self.n_actions,
                                        input_dims=self.input_dims,
                                        name=self.env_name+'_'+self.algo+'_q_next',
                                        chkpt_dir=self.chkpt_dir, hid_size=hid_size)

        self.observation_buffer = np.zeros((self.lookback_steps, self.input_dims[0]))

        if load:
            self.load_models()

    def get_action(self, observation, action_space):
        if (observation.shape) != self.observation_buffer.shape:
            self.observation_buffer[:-1] = self.observation_buffer[1:]
            self.observation_buffer[-1] = observation
        else:
            self.observation_buffer = observation

        if np.random.random() > self.epsilon:
            state = T.tensor([self.observation_buffer], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
