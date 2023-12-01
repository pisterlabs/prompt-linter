# encoding: utf-8

"""HTM Agent for "Cart-Pole-v0" environment from OpenAI Gym.
The system is controlled by applying a force of +1 or -1 to the cart. The
pendulum starts upright, and the goal is to prevent it from falling over. A
reward of +1 is provided for every time step that the pole remains upright. The
episode ends when the pole is more than 15 degrees from vertical, or the cart
moves more than 2.4 units from the center.

See https://github.com/openai/gym/wiki/CartPole-v0 and
https://gym.openai.com/envs/CartPole-v0"""

from __future__ import unicode_literals

import gym
import numpy as np
import random

from htm_learner import HtmLearner


def run(env, learner, numEpisodes, numSteps, render=False, sampleSize=500,
        verbosity=0):
    scores = []
    aveCumulativeReward = None
    for episode in range(numEpisodes):
        observation = env.reset()

        # HTM won't have an initial prediction, so choose one randomly to start?
        action = env.action_space.sample()
        currentState = learner.compute(observation, action)

        cumulativeReward = 0
        for t in range(numSteps):
            if render:
                env.render()

            action = learner.bestAction(currentState)
            observation, reward, done, info = env.step(action)

            cumulativeReward += learner.discount * reward
            newState = learner.compute(observation, action)
            learner.update(currentState, action, newState, reward)
            currentState = newState

            if done:
                learner.updateWhenDone(cumulativeReward, aveCumulativeReward)
                learner.reset()

                scores.append(t + 1)
                samples = min(sampleSize, len(scores))
                sampleScores = scores[-sampleSize:]

                if verbosity > 0:
                    print "Episode {}: {} steps".format(episode, t + 1)

                print u"Ave steps (n={}): {:.2f} ± {:.2f}".format(samples,
                    np.mean(sampleScores), np.std(sampleScores))
                print
                break


def main():
    randomSeed = 42
    random.seed(randomSeed)
    np.random.seed(randomSeed)

    envId = "CartPole-v0"
    env = gym.make(envId)

    alpha = 0.3
    epsilon = 0.75
    epsilonDecay = 0.99
    discount = 0.95
    k = 0.01
    learner = HtmLearner(env, alpha, epsilon, epsilonDecay, discount, k)

    numEpisodes = 500  # akin to trials

    # simulation steps per trial
    # CartPole-v0 defines "solving" as getting average reward of 195.0 over 100
    # consecutive trials.
    numSteps = 200
    run(env, learner, numEpisodes, numSteps)


if __name__ == "__main__":
    main()
