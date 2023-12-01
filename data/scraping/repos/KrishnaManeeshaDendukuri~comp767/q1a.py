#!/usr/bin/env python

import numpy as np
import gym
import argparse
from pathlib import Path
import os
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        help="Number of independent runs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--segments",
        help=(
            "Number of segments. Each segment there are 10 episodes "
            "of training, followed by 1 episode in which you simply "
            "run the optimal policy so far."
        ),
        type=int,
        default=100
    )

    parser.add_argument(
        "--temperature",
        help=(
            "Temperature value that will control the exploration. "
            "Higher value will increase exploration. Lower value "
            "will increase exploitation."
        ),
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--alpha",
        help="Learning rate",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--gamma",
        help="Discount factor",
        type=float,
        default=0.9
    )

    parser.add_argument(
        "-s", "--save",
        help="Path where to save all the files",
        default=Path('data/'),
        type=Path
    )

    parser.add_argument(
        "-m", "--method",
        help="Method to use",
        default="sarsa",
        choices=["sarsa", "expected_sarsa", "q_learning"]
    )

    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("Temperature needs to be greater than zero.")
    os.makedirs(args.save, exist_ok=True)
    return args


class Agent(object):
    """
    Agent running in the environment `Taxi-V2` from OpenAI.
    """

    def __init__(
        self,
        method,
        temperature=1.0,
        max_steps=100,
        alpha=0.1,
        gamma=0.9
    ):
        super(Agent, self).__init__()
        if method not in ['sarsa', 'expected_sarsa', 'q_learning']:
            raise ValueError("Method not supported")

        self.method = method
        self.temperature = temperature
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.env = gym.make("Taxi-v2")
        num_actions = self.env.action_space.n
        self.q_table = np.zeros(
            (self.env.observation_space.n, num_actions)
        )

    def run_episode(self, greedy=False):

        episode_reward = 0
        done = False

        # Initial state for the episode. State is a number, so we can
        # use it to index our q_table and policy
        state = self.env.reset()

        # At worst, will terminate at env._max_episode_steps.
        while not done:

            if greedy:
                action = randmax(self.q_table[state])
            else:
                action = softmax(self.q_table[state], self.temperature)

            s_prime, reward, done, _ = self.env.step(action)
            episode_reward += reward

            if self.method == 'q_learning':
                error = self.q_learning_error(s_prime, state, reward, action)

            elif self.method == 'sarsa':
                error = self.sarsa_error(s_prime, state, reward, action)

            elif self.method == 'expected_sarsa':
                error = self.exp_sarsa_error(s_prime, state, reward, action)

            # Don't update policy during greedy runs.
            if not greedy:
                # Update estimate of Q(s, a) using a small step size on error.
                self.q_table[state, action] += self.alpha * error

            # Set t+1 to t, for the next loop.
            state = s_prime

        return(episode_reward)

    def q_learning_error(self, s_prime, state, reward, action):
        """Q learning takes the max action from state s_prime."""
        a_prime = randmax(self.q[s_prime])

        error = reward
        error += self.gamma * self.q_table[s_prime, a_prime]
        error -= self.q_table[state, action]

        return(error)

    def sarsa_error(self, s_prime, state, reward, action):
        """Sarsa samples again from the policy to get a_prime."""
        a_prime = softmax(self.q_table[s_prime], self.temperature)

        error = reward
        error += self.gamma * self.q_table[s_prime, a_prime]
        error -= self.q_table[state, action]

        return(error)

    def exp_sarsa_error(self, s_prime, state, reward, action):
        """
        Takes a weighted average of all possible action from state s_prime.
        """
        probs = softmax(self.q_table[s_prime], self.temperature, sample=False)
        expectation = (probs * self.q_table[s_prime]).sum()

        error = reward
        error += self.gamma * expectation
        error -= self.q_table[state, action]

        return(error)


def softmax(action_values, temperature=1.0, sample=True):
    """
    Temperatures < 1 make the distribution encoded by action_values more
    "peaky", encouraging exploitation. Temperatures > 1 make the distribution
    more flat, encouraging exploration.
    """
    # Numerical stability (remove max after applying temperature).
    z = action_values / temperature
    z -= np.max(z)

    exp_val = np.exp(z)

    probs = exp_val / exp_val.sum()

    if sample:
        return(np.random.choice(len(probs), p=probs))
    else:
        return(probs)


def randmax(action_values):
    """
    Get the maximum element, breaking ties randomly.
    """
    maxes = np.flatnonzero(action_values == np.max(action_values))
    action = np.random.choice(maxes)

    return(action)


if __name__ == "__main__":
    args = parse_args()

    N_EPISODES = 10

    methods = ["sarsa", "expected_sarsa", "q_learning"]
    alphas = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    temps = np.array([0.1, 1.0, 10.0])

    test_rewards = np.zeros((len(methods),
                             len(alphas),
                             len(temps),
                             args.segments,
                             args.runs))

    train_rewards = np.zeros_like(test_rewards)

    for i, method in enumerate(tqdm.tqdm(methods, desc="methods")):
        for j, alpha in enumerate(tqdm.tqdm(alphas, desc="alpha")):
            for k, temp in enumerate(tqdm.tqdm(temps, desc="temp")):
                for run in tqdm.trange(args.runs, desc="run"):

                    agent = Agent(
                        method=args.method,
                        alpha=alpha,
                        temperature=temp,
                        gamma=args.gamma)

                    for segment in tqdm.trange(args.segments, desc="segment"):

                        for episode in range(N_EPISODES):
                            reward = agent.run_episode()
                            train_rewards[i, j, k, segment, run] += reward

                        # Store mean training performance over N_EPISODES.
                        train_rewards[i, j, k, segment, run] /= N_EPISODES

                        # Store test performance following greedy policy.
                        reward = agent.run_episode(greedy=True)
                        test_rewards[i, j, k, segment, run] = reward

    results = {'train': train_rewards, 'test': test_rewards}
    np.save(os.path.join(args.save, "q1a.npy"), results)
