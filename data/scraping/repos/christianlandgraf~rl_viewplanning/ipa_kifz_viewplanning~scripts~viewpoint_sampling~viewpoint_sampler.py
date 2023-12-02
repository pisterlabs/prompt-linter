#!/usr/bin/env python
import os.path
import gym
import numpy as np
import pandas as pd
import time
import datetime
from gym import wrappers
import click
import csv

# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

if __name__ == '__main__':

    rospy.init_node('ipa_kifz_viewpoint_sampling',
                    anonymous=True,
                    log_level=rospy.WARN)

    # Init OpenAI_ROS Environment
    rospy.loginfo("Init Task Environment")
    task_and_robot_environment_name = rospy.get_param(
        '/ipa_kifz/task_and_robot_environment_name')
    rospy.loginfo("Init ROS Environment")
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Viewpoint sampling")

    # Create Viewpoint file and add column header
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ipa_kifz_viewplanning')
    viewpoint_filename = os.path.join(pkg_path, "config",
                                      "ipa_kifz_viewpoints.csv")

    if not os.path.isfile(viewpoint_filename):
        with open(viewpoint_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

# Start viewpoint generation
    for i in range(1):
        observation = env.reset()
        time.sleep(1)

        # Execute the action in the environment and get feedback
        for i in range(2000):
            observation, reward, done, info = env.step(0)
            print(observation[:7])
            if click.confirm('Do you want to save this viewpoint?',
                             default=True):
                print("Save viewpoint")
                with open(viewpoint_filename, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        observation[0], observation[1], observation[2],
                        observation[3], observation[4], observation[5],
                        observation[6]
                    ])
            else:
                print("Go to next viewpoint")
    env.close()
