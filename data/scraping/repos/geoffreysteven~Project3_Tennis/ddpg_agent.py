import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

    def learn_batch(self):
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # as per class benchmark implementation Attempt #3
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = 1e-2
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self): # based on brownian motion as per class code sample
        """Update internal state and return it as a noise sample."""
        x = self.state
        
        #### - Old code from DDPG Pendulum
        # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])

        ### - Revised OUNoise from OpenAI Noise Blog/Paper
        # replace with AI Gym's https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py:
        # x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.array([np.random.normal() for i in range(len(x))])
        self.state = x + dx
        return self.state

    # def sample(self): # normal gaussian noise
    #     """Update internal state and return it as a noise sample."""
    #     x = self.state
    #     dx = return np.random.normal(self.mu, self.sigma)
    #     self.state = x + dx
    #     return self.state

# OrnsteinUhlenbeckActionNoise is from OpenAI's blog on noise:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
# The noise code below is from OpenAI's blog, copied here for reference

# class ActionNoise(object):
#     def reset(self):
#         pass
#
# class OrnsteinUhlenbeckActionNoise(ActionNoise):
#     def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
#         self.theta = theta
#         self.mu = mu
#         self.sigma = sigma
#         self.dt = dt
#         self.x0 = x0
#         self.reset()
#
#     def __call__(self):
#         x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
#         self.x_prev = x
#         return x
#
#     def reset(self):
#         self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
#
#     def __repr__(self):
#         return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# AdaptiveParamNoiseSpec is from OpenAI's blog on noise:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py

# class AdaptiveParamNoiseSpec(object):
#     def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
#         self.initial_stddev = initial_stddev
#         self.desired_action_stddev = desired_action_stddev
#         self.adoption_coefficient = adoption_coefficient
#
#         self.current_stddev = initial_stddev
#
#     def adapt(self, distance):
#         if distance > self.desired_action_stddev:
#             # Decrease stddev.
#             self.current_stddev /= self.adoption_coefficient
#         else:
#             # Increase stddev.
#             self.current_stddev *= self.adoption_coefficient
#
#     def get_stats(self):
#         stats = {
#             'param_noise_stddev': self.current_stddev,
#         }
#         return stats
#
#     def __repr__(self):
#         fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
#         return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)

# NormalActionNoise is from OpenAI's blog on noise:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py

# class NormalActionNoise(ActionNoise):
#     def __init__(self, mu, sigma):
#         self.mu = mu
#         self.sigma = sigma
#
#     def __call__(self):
#         return np.random.normal(self.mu, self.sigma)
#
#     def __repr__(self):
#         return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# code for replaybufer is from DDPG_Pendulum class repo
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
