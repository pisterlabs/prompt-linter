from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys;sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os

import argparse
import time
import logging
import json
import gym
import plotly
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from tensorforce import TensorForceError
from tensorforce.execution import Runner
from tensorforce.agents import DQNAgent
from tensorforce.agents import Agent
from tensorforce.core.explorations import EpsilonDecay

from pycrysfmlEnvironment import PycrysfmlEnvironment

from tensorforce.contrib.openai_gym import OpenAIGym

#Based on quickstart and example code from TensorForce documentation

#@misc{schaarschmidt2017tensorforce,
#    author = {Schaarschmidt, Michael and Kuhnle, Alexander and Fricke, Kai},
#    title = {TensorForce: A TensorFlow library for applied reinforcement learning},
#    howpublished={Web page},
#    url = {https://github.com/reinforceio/tensorforce},
#    year = {2017}
#}

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--agent-config', help="Agent configuration file")
    args = parser.parse_args()

    #From quickstart on docs
    #Network as list of layers
    #This is from mlp2_embedding_network.json
    network_spec = [
        {
            "type": "dense",
            "size":  32
#            "activation": "relu"
        },
        {
            "type": "dense",
            "size": 32
#            "activation": "relu"
        }
    ]

    DATAPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    observedFile = os.path.join(DATAPATH,r"prnio.int")
    infoFile = os.path.join(DATAPATH,r"prnio.cfl")

    environment = PycrysfmlEnvironment(observedFile, infoFile)

    #get agent configuration
    if args.agent_config is not None:
        with open(args.agent_config, 'r') as fp:
            agent_config = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    agent = Agent.from_spec(
            spec=agent_config,
            kwargs=dict(
                states=environment.states,
                actions=environment.actions,
                network=network_spec,
            )
        )

    #Use this line to resore a pre-trained agent
    #agent.restore_model(file="/mnt/storage/deepQmodel_chisq")

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    rewardsLog = []
    steps = []

    def episode_finished(r):

        if r.episode % 10 == 0:
            rewardsLog.append(r.episode_rewards[-1])
            steps.append(r.episode)

        if r.episode % 50 == 0:
            sps = r.timestep / (time.time() - r.start_time)
            file = open("/mnt/storage/trainingLog", "a")
            file.write("Finished episode {ep} after {ts} timesteps. Steps Per Second {sps}\n".format(ep=r.episode,
                                                                                                    ts=r.timestep,
                                                                                                    sps=sps))
            file.write("Episode reward: {}\n".format(r.episode_rewards[-1]))
            file.write("Episode timesteps: {}\n".format(r.episode_timestep))
            file.write("Average of last 500 rewards: {}\n".format(sum(r.episode_rewards[-500:]) / 500))
            file.write("Average of last 100 rewards: {}\n".format(sum(r.episode_rewards[-100:]) / 100))

            agent.save_model(directory="/mnt/storage/deepQmodel_simpleA_stdreward", append_timestep=False)

        return True

    runner.run(
        timesteps=60000000,
        episodes=5000,
        max_episode_timesteps=1000,
        deterministic=False,
        episode_finished=episode_finished
    )

    #graph rewards
    plt.scatter(steps, rewardsLog)
    plt.savefig('/mnt/storage/rewardLog_simpleA_stdreward.png')

    runner.close()

if __name__ == '__main__':
    main()

