"""
Implement vanilla policy gradient algorithm. This code is constructed with
inspiration from OpenAI Spinning-Up RL. All credits go to them.

Vanilla Policy Gradient includes the following steps:
- Run the current policy for a fixed amount of timestep (get info for a single trajectory)
    + In each timestep, the agent takes in a state and returns a probability
    distribution of actions
    + Sample an action from that action distribution and interact with the
    environment
    + Record the reward, save the (state, action, reward, state_next) tuples
    + Doing this for fixed amount of timesteps and record the total reward
- Follow the above procedure for several times (get info for several trajectories)
- Estimate the policy gradient, given the reward for each trajectory, the action
distribution of each timestep in each trajectory
- Update the parameters using gradient ascent.
"""
import imageio
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


EPOCHS = 1000
SAMPLES_PER_EPOCH = 5000


def get_agent():
    """Get the agent that works for the CartPole-v1 environment
    """

    return nn.Sequential(
        nn.Linear(in_features=4, out_features=32),
        nn.Tanh(),
        nn.Linear(in_features=32, out_features=64),
        nn.Tanh(),
        nn.Linear(in_features=64, out_features=2),
        nn.Softmax()
    )


if __name__ == '__main__':
    """Run the training"""

    # Initialize the agent, environment and the training procedure
    env = gym.make('CartPole-v1')
    agent = get_agent()
    optimizer = optim.Adam(params=agent.parameters(), lr=1e-3)

    avg = 0
    counter = 0
    for epoch in range(EPOCHS):

        # report
        report_total_rewards = []
        report_n_episodes = 0

        epoch_observations = []
        epoch_rewards = []
        epoch_sampled_actions = []

        # information to retrieve for each episode
        observation = env.reset()
        eps_reward = []
        done = False

        while True:

            observation = torch.FloatTensor(observation)
            epoch_observations.append(observation)

            # get the action
            action_prob = agent(observation)
            action = torch.distributions.Categorical(action_prob).sample().item()
            epoch_sampled_actions.append(action)

            # run the action on the environment
            observation, reward, done, info = env.step(action)
            eps_reward.append(reward)

            if done:
                total_reward, reward_amount = sum(eps_reward), len(eps_reward)
                epoch_rewards += [total_reward] * reward_amount
                # @NOTE: in better implementation, epoch_rewards is not the
                # the total rewards, but instead the advantage values of the
                # corresponding action. -> we get the advantage, and we have
                # the action distribution, which is enough for estimate the
                # gradient of the total reward wrt to policy parameters:
                # \delta log(pi_\theta(action_t | observation)) * Advantage)

                report_total_rewards.append(total_reward)
                report_n_episodes += 1

                if len(epoch_observations) > SAMPLES_PER_EPOCH:
                    break

                observation, eps_reward, done = env.reset(), [], False

        # perform optimization here
        observation = TensorDataset(
            torch.stack(epoch_observations),
            torch.FloatTensor(epoch_rewards),
            torch.LongTensor(epoch_sampled_actions))
        observation = DataLoader(observation, batch_size=500)

        for batch_observation, batch_reward, batch_actions in observation:
            mini_batch = batch_observation.size(0)

            # only interest in the taken actions (because the taken actions
            # from the trajectory - non-taken actions do not contribute to
            # that specific trajectory hence will be left out)
            masked_actions = torch.zeros(mini_batch, 2)
            masked_actions[torch.arange(mini_batch), batch_actions] = 1

            # get action log probability multiplied by the reward (or advantage)
            actions = agent(batch_observation)
            actions = -(torch.log(actions)
                * masked_actions
                * batch_reward.unsqueeze(1)).mean()

            # perform optimization
            optimizer.zero_grad()
            actions.backward()
            optimizer.step()

            counter += 1
            avg = (avg * (counter - 1) + actions.item()) / (counter + 1)

        print('[{:4d}]: {}'.format(epoch + 1, sum(report_total_rewards) / report_n_episodes))

        # perform evaluation
        images = []
        observation = env.reset()
        image = env.render('rgb_array')
        images.append(image)

        while True:
            observation = torch.FloatTensor(observation)

            action_prob = agent(observation)
            action = torch.distributions.Categorical(action_prob).sample().item()
            observation, reward, done, info = env.step(action)

            image = env.render('rgb_array')
            images.append(image)

            if done:
                imageio.mimsave(
                    'images/vpg_{:04d}.gif'.format(epoch+1),
                    images,
                    fps=20)
                break
