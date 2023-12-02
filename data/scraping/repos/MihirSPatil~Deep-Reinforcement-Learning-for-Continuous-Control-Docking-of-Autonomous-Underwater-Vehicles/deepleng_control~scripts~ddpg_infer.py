#!/usr/bin/env python3

import gym
from gym import wrappers
import rospy
import rospkg
import random
import torch
import numpy as np
from collections import deque
from ddpg_agent import Agent
# import our training environment
from openai_ros.task_envs.deepleng import deepleng_docking


class DdpgInfer():
    def __init__(self):
        self.env = gym.make('DeeplengDocking-v1')
        rospy.loginfo("Gym environment done")
        self.agent = Agent(state_size=13, action_size=3, random_seed=2)
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('deepleng_control')
        self.outdir = pkg_path + '/training_results'
        self.agent.actor_local.load_state_dict(torch.load(self.outdir + '/checkpoint_actor.pth'))
        self.agent.critic_local.load_state_dict(torch.load(self.outdir + '/checkpoint_critic.pth'))

    def __call__(self, *args, **kwargs):

        state = self.env.reset()
        for t in range(500):
            action = self.agent.act(state, add_noise=False)
            # env.render()
            state, reward, done, _ = self.env.step(action)
            print("state:", state)
            print("Reward: ", reward)
            if done:
                break

        self.env.close()

def main():
    rospy.init_node('DdpgDeepleng_infer', anonymous=True)
    infer = DdpgInfer()
    infer()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")

if __name__ == '__main__':
    main()
