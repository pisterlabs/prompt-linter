#!/usr/bin/env python

import numpy as np
import gym
import argparse
from pathlib import Path
import os
import tqdm
from rlai import IHT, tiles
from gym.envs.classic_control import PendulumEnv

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        help="Number of independant runs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--episodes",
        help=(
            "Number of segments. Each one will start at (0, 0)"
        ),
        type=int,
        default=200
    )

    parser.add_argument(
        "--torque_prob",
        help=(
            "produces torque in the same direction as the current velocity "
            "with probability p and in the opposite direction with "
            "probability (1-p). If velocity is 0, you can torque in a "
            "random direction."
        ),
        type=float,
        default=0.9
    )

    parser.add_argument(
        "--tiling",
        help=(
            "Number of overlapping tiling for the discretization of "
            "the angular position and angular velocity"
            "NOTE: The number of overlapping tilings will divide the "
            "learning rate."
        ),
        type=int,
        default=5
    )

    parser.add_argument(
        "--trace_decay",
        help="Trace decay",
        default=[0, 0.3, 0.7, 0.9, 1],
        type=float,
        nargs="*"
    )

    parser.add_argument(
        "--gamma",
        help="Discount factor",
        default=0.9,
        type=float
    )

    parser.add_argument(
        "--alpha",
        help=(
            "Learning rate"
            "NOTE: The learning rate will be divided by the number of "
            "overlapping tilings."
        ),
        type=float,
        default=[1 / 4, 1 / 8, 1 / 16],
        nargs="*"
    )

    args = parser.parse_args()

    return args


#def monkey_reset(self):
#    self.state = np.array([np.pi / 2, 0])
#    self.last_u = None
#    return self._get_obs()


class Agent(object):
    """
    Agent running in the environment `Taxi-V2` from OpenAI.
    """

    def __init__(self, alpha=0.1, gamma=0.9, trace_decay=0.9,
                 torque_prob=0.9, tilings=5, seed=1234):

        super(Agent, self).__init__()

        self.alpha = alpha / tilings
        self.gamma = gamma
        self.trace_decay = trace_decay
        self.torque_prob = torque_prob
        self.tilings = tilings

        #PendulumEnv.reset = monkey_reset
        self.env = PendulumEnv()
        self.env.seed(seed)
        self.n_tiles = self.tilings * 10 * 10

        np.random.seed(seed)
        self.eligibility = np.zeros(self.n_tiles)
        self.weights = np.random.uniform(-0.001, 0.001, size=self.n_tiles)

    def get_tile_positions(self, angle, velocity):
        """
        Given an angle, velocity, and hashtable, find the corresponding
        indices in the tilings. Return a binary vector that is 1 at these
        locations.
        """
        angle_scaler = 10.0 / 2  # rescale from (-1, 1) to (0, 10)
        velocity_scaler = 10.0 / 16  # rescale from (-8, 8) to (0, 10)
        state = [(angle + 1) * angle_scaler, (velocity + 8) * velocity_scaler]

        idx = tiles(self.iht, self.tilings, state)
        x = np.zeros(self.n_tiles)
        x[idx] = 1

        return(x)

    def run_episode(self):

        # We want one extra cell in each dimension of IFT, hense 11 * 11.
        self.iht = IHT(self.tilings * 11 * 11)
        self.eligibility = np.zeros(self.n_tiles)
        done = False
        i = 0

        # Reset the pendulum to be upright.
        # state[0] = cos(theta), ranges from -1 to 1
        # state[1] = sin(theta), ranges from -1 to 1
        # state[2] = angular velocity , ranges from -8 to 8
        state = self.env.reset()
        self.env.state = np.array([0, 0])
        self.env.last_u = None
        state = np.array([1, 0, 0])

        while not done:

            # This is the input to the value function
            x = self.get_tile_positions(state[1], state[2])

            # Fixed policy: produces torque in the same direction as the
            # current velocity with p=0.9, opposite direction p=0.1.
            # If velocity is 0, you can torque in a random direction.
            velocity_direction = np.sign(state[2])
            rand = np.random.rand()
            if (state[2] == 0 and rand < 0.5) or rand > self.torque_prob:
                velocity_direction *= -1

            # Randomly sample torque (action) from [-2 2] (Sample from 0 to 2
            # and multiply by the velocity_direction).
            action = np.random.uniform(0, 2) * velocity_direction

            # Take action according to our policy.
            s_prime, reward, done, _ = self.env.step([action])

            # This returns a list of {self.tilings} integers. Those are the
            # states we want to update.
            x_prime = self.get_tile_positions(s_prime[1], s_prime[2])

            # NB: grad of v(S,w) == x b/c this is linear function approx.
            self.eligibility *= self.trace_decay * self.gamma
            self.eligibility += x

            delta = reward + self.gamma * np.dot(self.weights, x_prime)
            delta -= np.dot(self.weights, x)

            # Update the weights
            self.weights += self.alpha * delta * self.eligibility

            # Set t+1 to t, for the next loop.
            state = s_prime
            i += 1
            if i > 200:
                done = True

        # Task is to plot the value of state (0, 0)
        start_state = self.get_tile_positions(0, 0)

        return(np.dot(self.weights, start_state))


if __name__ == '__main__':
    args = parse_args()
    print(args)
    seeds = list(range(42, 42 + args.runs))
    values = np.zeros(
        (args.runs, args.episodes, len(args.trace_decay), len(args.alpha))
    )

    for run in tqdm.trange(args.runs, desc='Run'):
        for i, trace_decay in enumerate(tqdm.tqdm(args.trace_decay, desc="Trace decay")):
            for j, alpha in enumerate(tqdm.tqdm(args.alpha, desc="Alpha")):

                agent = Agent(
                    alpha=alpha,
                    gamma=args.gamma,
                    trace_decay=trace_decay,
                    torque_prob=args.torque_prob,
                    tilings=args.tiling,
                    seed=seeds[run]
                )

                for episode in tqdm.trange(args.episodes, desc="Episode"):
                    start_state_value = agent.run_episode()
                    values[run, episode, i, j] = start_state_value

    np.save("data/q2a.npy", values)
