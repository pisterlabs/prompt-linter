#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The monitor node.
"""

import rospy
import utils
import unittest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from simulation.srv import StairInfo, StairInfoRequest
from simulation.srv import GoalInfo, GoalInfoRequest, GoalInfoResponse
from simulation.srv import StairInfo, StairInfoRequest, StairInfoResponse
from monitor.srv import NewRollout, NewRolloutRequest, NewRolloutResponse
from monitor.srv import StepReturn, StepReturnRequest, StepReturnResponse
from monitor.msg import RolloutAnalytics
from monitor.srv import GuidanceInfo, GuidanceInfoResponse, GuidanceInfoRequest
from simulation.msg import DistDirec
from control.msg import State
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, Pose, Point
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_msgs.msg import Float32

import tf
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np


class TestMonitor:
    def __init__(self):
        rospy.init_node('test_monitor')
        rospy.set_param("exp_series_name", "exp_monitor_test")
        # Subscribers
        self.robot_state = rospy.Publisher("/robot/state", State, queue_size=1)
        self.safety_deviation = rospy.Publisher("/safety/relative_deviation", Float32, queue_size=1)
        self.safety_angular = rospy.Publisher("/safety/angular", Float32, queue_size=1)
        self.odometry = rospy.Publisher("/odometry", Odometry, queue_size=1)
        # Publishers
        self.rollout_analytics = rospy.Subscriber("/rollout/analytics", RolloutAnalytics, self.callback_rollout_analytics, queue_size=1)
        # Clients
        _ = rospy.Service("/goal_info", GoalInfo,  self.callback_goal_info)
        _ = rospy.Service("/stair_info", StairInfo, self.callback_stair_info)
        # Services
        self.rollout_new = rospy.ServiceProxy("/rollout/new", NewRollout)
        self.rollout_step = rospy.ServiceProxy("/rollout/step_return", StepReturn)
        self.rollout_start = rospy.ServiceProxy("/rollout/start", Trigger)
        self.rollout_guidance = rospy.ServiceProxy("/guidance/info", GuidanceInfo)
        # data
        self.goal_info = None
        self.stair_info = None

        self.results = []

    def callback_rollout_analytics(self, req):
        self.results.append(req)

    def callback_goal_info(self, req):
        return self.goal_info

    def callback_stair_info(self, req):
        return self.stair_info

    def iterate_over_publishers(self):
        self.robot_state.publish(State(
            linear=0.3,
            angular=0.2,
            front_flippers=0.1,
            rear_flippers=-0.1,
            arm_joint1=0.5,
            arm_joint2=0.2,
        ))
        self.safety_deviation.publish(Float32(data=0.5))
        self.safety_angular.publish(Float32(data=0.5))
        self.odometry.publish(Odometry(
            pose=PoseWithCovariance(
                pose=Pose(
                    position=Point(
                        x=self.goal_info.x,
                        y=self.goal_info.y,
                        z=self.goal_info.z
                    )
                )
            )
        ))
        rospy.sleep(0.01)

    def init_rollout(self, rollout):
        _ = self.rollout_new.call(rollout)
        self.iterate_over_publishers()
        err = True
        while err:
            try:
                _ = self.rollout_start.call(TriggerRequest())
                self.iterate_over_publishers()
                err = False
            except Exception:
                print("Err")

    def test_full(self):
        rollout = NewRolloutRequest(
            experiment="full",
            seq=1,
            time_step_limit=1,
            sensors="",
            complexity_type="full",
            arm=True,
            angular=False,
            use_penalty_angular=False,
            use_penalty_deviation=False,
        )
        self.goal_info = GoalInfoResponse(
            task="ascent",
            rand=False,
            x=1.0,
            y=1.0,
            z=1.0,
        )
        self.stair_info = StairInfoResponse(
            length=0.1,
            height=0.1,
            number=2,
            exist=True,
        )
        rollout.seq += 1
        new_episode = True
        while True:
            if new_episode:
                self.init_rollout(rollout)
                new_episode = False
            self.iterate_over_publishers()
            resp = self.rollout_step.call(StepReturnRequest())
            step_reward = resp.reward
            print("Step reward", step_reward)
            episode_done = resp.done
            if episode_done:
                resp = self.rollout_guidance.call(GuidanceInfoRequest())
                level = resp.level
                epsilon = resp.epsilon
                global_done = resp.done
                print(f"Done episode {epsilon}, {level}")
                new_episode = True
                if global_done:
                    break


if __name__ == '__main__':
    TestMonitor().test_full()