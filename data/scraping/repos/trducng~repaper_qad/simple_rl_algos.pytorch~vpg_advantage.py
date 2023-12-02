"""
Implement the vanilla policy gradient algorithm that employs the advantage
function as a scaler for the right hand side of the policy gradient estimation.
The overall idea should be the same in `vpg.py`, with only the replacement of
total reward with advantage value. This code is constructed with inspiration
from OpenAI Spinning-Up RL. All credits go to them.

When using the advantage as a baseline function, we will need:
- Action value function: this function can be obtained from the trajectory
- Value function: this should be approximate (maybe using monte-carlo, or using
a separate neural network like actor-critic, or using TD-learning)

In this document: https://spinningup.openai.com/en/latest/algorithms/vpg.html,
the value function is parameterized, and fit with the reward-to-go using mean-
squared error. When doing so, the value function value will approximate its
reward-to-go version, then in the baseline policy gradient, we have:
    \delta = E [ \delta log(pi|s_t) * RTG(s_t) ]
    -> \delta = E [ \delta log(pi|s_t) * (RGG(s_t) - V(s_t))]
    -> \delta = E [ \delta log(pi|s_t) * 0 ]   -> which is nonsense

It seems easy to incorporate GAE into the policy gradient estimation.
"""
import imageio
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


EPOCHS = 1000
SAMPLES_PER_EPOCH = 5000
GAMMA = 0.99
LAMBDA = 0.97


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


def get_value_approximator():
    """Get the value function approximator"""

    return nn.Sequential(
        nn.Linear(in_features=4, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=1)
    )


def reward_to_go(rewards):
    """Calculate rewards to go. This reward to go will serve as Q value

    This RTG function also employs lambda and gamma coefficient from the GAE
    paper.

    [r1, xr1 + r2, xxr1 + xr2 + r3, xxxr1 + xxr2 + xr3 + r4]
    """
    new_rewards = []
    for idx, each_reward in enumerate(reversed(rewards)):
        if idx == 0:
            new_rewards.append(each_reward)
            continue
        new_rewards.append(GAMMA * LAMBDA * new_rewards[idx - 1] + each_reward)

    return list(reversed(new_rewards))


if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    agent = get_agent()
    valuer = get_value_approximator()
    loss_value = nn.MSELoss()

    optimizer_agent = optim.Adam(params=agent.parameters(), lr=1e-3)
    optimizer_value = optim.Adam(params=valuer.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):

        epoch_observations = []
        epoch_sampled_actions = []
        epoch_rewards = []

        # for episode report purpose
        report_total_rewards = []
        report_n_episodes = 0

        # each episode information
        observation = env.reset()
        eps_reward = []
        done = False

        agent.eval()
        while True:

            # collect the experience: observation->action->observation, reward
            observation = torch.FloatTensor(observation)
            epoch_observations.append(observation)

            action_prob = agent(observation)
            action = torch.distributions.Categorical(action_prob).sample().item()
            epoch_sampled_actions.append(action)

            observation, reward, done, info = env.step(action)
            eps_reward.append(reward)

            if done:
                # we want reward-to-go
                epoch_rewards += reward_to_go(eps_reward)

                # reporting
                report_total_rewards.append(sum(eps_reward))
                report_n_episodes += 1

                if len(epoch_observations) > SAMPLES_PER_EPOCH:
                    break

                observation, eps_reward, done = env.reset(), [], False

        # do policy gradient
        # expectation of \delta(log policy) * (RTG - V(s_t))
        experience = TensorDataset(
            torch.stack(epoch_observations),
            torch.FloatTensor(epoch_rewards),
            torch.LongTensor(epoch_sampled_actions)
        )
        experience = DataLoader(experience, batch_size=500)

        agent.train()
        for batch_observations, batch_rewards, batch_actions in experience:
            mini_batch = batch_observations.size(0)
            batch_rewards = batch_rewards.unsqueeze(1)

            # get the state value
            state_value = valuer(batch_observations)
            loss = loss_value(state_value, batch_rewards)

            # update the agent
            masked_actions = torch.zeros(mini_batch, 2)
            masked_actions[torch.arange(mini_batch), batch_actions] = 1

            # get action log probability
            actions = agent(batch_observations)
            actions = - (torch.log(actions) * masked_actions
                    * (batch_rewards - state_value)).mean()

            # perform policy optimization
            if actions.item() > 0:
                # if actions is negative, then the parameter gradient point to
                # direction of decreasing reward value
                optimizer_agent.zero_grad()
                actions.backward(retain_graph=True)
                optimizer_agent.step()

            # perform value optimization
            optimizer_value.zero_grad()
            loss.backward()
            optimizer_value.step()

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
                    'images/advantage_{:04d}.gif'.format(epoch+1),
                    images,
                    fps=20)
                break