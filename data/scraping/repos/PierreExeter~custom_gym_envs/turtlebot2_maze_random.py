#!/usr/bin/env python

import gym
import rospy
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

# initialise environment
rospy.init_node('turtlebot2_maze_random', anonymous=True, log_level=rospy.WARN)
task_and_robot_environment_name = rospy.get_param('/turtlebot2/task_and_robot_environment_name')
env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

print("Environment: ", env)
print("Action space: ", env.action_space)
# print(env.action_space.high)
# print(env.action_space.low)
print("Observation space: ", env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


for episode in range(20):

    env.reset()

    for t in range(100):

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print("episode: ", episode)
        print("timestep: ", t)
        print("obs: ", obs)
        print("action:", action)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)
        
        if done:
            print("Episode {} finished after {} timesteps".format(episode, t+1))
            break

env.close()