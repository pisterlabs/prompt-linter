#!/usr/bin/env python
"""Mario Gym Adventure!

python mario.py ppaquette/meta-SuperMarioBros-v0 \
  -a DQFDAgent -c mario_agent.json \
  -ld ./demos/ -s ./agents/ -m ./monitors/ -mv 1000 -D
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

import demo

from model import mario_net

from ppaquette_gym_super_mario import wrappers

from tensorforce import Configuration, TensorForceError
from tensorforce.agents import agents
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.execution import Runner


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="ID of the gym environment, i.e. ppaquette/SuperMarioBros-1-1-v0")
    parser.add_argument('-a', '--agent', help='Agent')
    parser.add_argument('-c', '--agent-config', help="Agent configuration file")
    parser.add_argument('-e', '--episodes', type=int, default=50000, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=100000, help="Maximum number of timesteps per episode")
    parser.add_argument('-m', '--monitor', help="Save results to this directory")
    parser.add_argument('-ms', '--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('-mv', '--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")
    parser.add_argument('-ld', '--load-demo', required=True, help="Load demos from this dir")
    parser.add_argument('-pt', '--pretrain', action='store_true', default=False, help="Pretrain agent on demos")
    parser.add_argument('-ul', '--use_lstm', action='store_true', default=False, help="Use LSTM model")
    parser.add_argument('-ls', '--lstm_size', type=int, default=256, help="LSTM size")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    environment = OpenAIGym(args.gym_id,
                            monitor=args.monitor,
                            monitor_safe=args.monitor_safe,
                            monitor_video=args.monitor_video)
    mode_wrapper = wrappers.SetPlayingMode('algo')
    ac_wrapper = wrappers.ToDiscrete()
    environment.gym = mode_wrapper(ac_wrapper(environment.gym))

    if args.agent_config:
        agent_config = Configuration.from_json(args.agent_config)
    else:
        agent_config = Configuration()
        logger.info("No agent configuration provided.")

    agent_config.default(dict(states=environment.states,
                              actions=environment.actions,
                              network=mario_net(name='mario',
                                                lstm_size=args.lstm_size,
                                                actions=environment.actions['num_actions'],
                                                use_lstm=args.use_lstm)))
    agent = agents[args.agent](config=agent_config)

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        logger.info("-" * 16)
        agent.load_model(args.load)
        logger.info("Loaded {}".format(agent))

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent_config)

    if args.save:
        save_dir = os.path.dirname(args.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))

    try:
        if args.load_demo:
            logger.info("-" * 16)
            logger.info("Loading demos")
            demos = demo.load(args.load_demo)
            logger.info("Importing demos")
            agent.import_demonstrations(demos)

            if args.pretrain:
                logger.info("-" * 16)
                logger.info("Pretraining {} steps".format(len(demos)))
                agent.pretrain(steps=len(demos))

        runner = Runner(
            agent=agent,
            environment=environment,
            repeat_actions=1,
            save_path=args.save,
            save_episodes=args.save_episodes
        )

        report_episodes = args.episodes // 1000
        if args.debug:
            report_episodes = 1

        def episode_finished(r):
            if r.episode % report_episodes == 0:
                logger.info("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep))
                logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
                logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
                logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
            return True

        logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))
        runner.run(args.episodes, args.max_timesteps, episode_finished=episode_finished)
        logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode))
    except (KeyboardInterrupt):
        agent.save_model(args.save)
        pass

    if args.monitor:
        environment.gym.monitor.close()
    environment.close()


if __name__ == '__main__':
    main()
