# -*- coding: utf-8 -*-
"""Train or test baselines on LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import importlib
import os
import rospy
import gym
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import algorithms.common.helper_functions as common_utils

# configurations
parser = argparse.ArgumentParser(description="Pytorch RL baselines")
parser.add_argument(
    "--seed", type=int, default=777, help="random seed for reproducibility"
)
parser.add_argument("--algo", type=str, default="sac", help="choose an algorithm")
parser.add_argument(
    "--load-from",
    type=str,
    default=None,
    help="load the saved model and optimizer at the beginning",
)
parser.add_argument("--episode-num", type=int, default=1500, help="total episode num")
parser.add_argument(
    "--max-episode-steps", type=int, default=76, help="max episode step"
)
parser.add_argument(
    "--off-render", dest="render", action="store_false", help="turn off rendering"
)
parser.add_argument(
    "--render-after",
    type=int,
    default=0,
    help="start rendering after the input number of episode",
)
# parser.add_argument(
#     "--use-se", dest="render", action="store_false", help="use self-explainer to help training"
# )

parser.add_argument("--save-period", type=int, default=100, help="save model period")
parser.add_argument("--log", action="store_true", help="turn on logging")
parser.add_argument("--test", action="store_true", help="test mode (no training)")
parser.add_argument(
    "--demo-path",
    type=str,
    default="data/lunarlander_continuous_demo.pkl",
    help="demonstration path",
)
parser.set_defaults(render=True)

args = parser.parse_args()


def main():
    """Main."""
    # ros env initialization
    # os.environ["ROS_MASTER_URI"] = "10.153.71.29:11311"
    # os.environ["ROS_IP"] = "yzha3@10.218.108.88"
    rospy.init_node('fetch_serl_train',
                    anonymous=True, log_level=rospy.WARN)

    # openai_ros env initialization
    task_and_robot_environment_name = 'FetchSERL-v3'
    print("AAA", task_and_robot_environment_name)
    rospy.logwarn("task_and_robot_environment_name ==>" +
                  str(task_and_robot_environment_name))
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    env._max_episode_steps = args.max_episode_steps

    env = gym.wrappers.Monitor(
        env, '../data/fetch-1', force=True)    # set a random seed
    common_utils.set_random_seed(args.seed, env)

    # run
    module_path = "config.agent.fetch_serl_insert_v0." + args.algo
    agent = importlib.import_module(module_path)
    agent = agent.get(env, args)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()


if __name__ == "__main__":
    main()
