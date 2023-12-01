#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The monitor node.
"""

import rospy
import utils
from guidance import Guidance
from std_srvs.srv import Trigger, TriggerResponse
from simulation.srv import StairInfo, StairInfoRequest
from simulation.srv import GoalInfo, GoalInfoRequest, GoalInfoResponse
from simulation.srv import StairInfo, StairInfoRequest, StairInfoResponse
from monitor.srv import NewRollout, NewRolloutRequest, NewRolloutResponse
from monitor.srv import StepReturn, StepReturnRequest, StepReturnResponse
from monitor.msg import RolloutAnalytics
from monitor.srv import GuidanceInfo, GuidanceInfoResponse
from simulation.msg import DistDirec
from control.msg import State
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_msgs.msg import Float32

import tf
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np


class Monitor:
    def __init__(self):
        rospy.init_node('monitor')
        # Subscribers
        rospy.Subscriber("/robot/state", State, self.callback_robot_state)
        rospy.Subscriber("/safety/relative_deviation", Float32, self.callback_safety_deviation)
        rospy.Subscriber("/safety/angular", Float32, self.callback_safety_angular)
        rospy.Subscriber("/odometry", Odometry, self.callback_odometry)
        # Publishers
        self.rollout_analytics = rospy.Publisher("/rollout/analytics", RolloutAnalytics)
        # Clients
        self.goal_info = rospy.ServiceProxy("goal_info", GoalInfo)
        self.stair_info = rospy.ServiceProxy("stair_info", StairInfo)
        # Services
        self.rollout_state = utils.RolloutState()
        s = rospy.Service("/rollout/new", NewRollout, self.callback_new_rollout)
        s = rospy.Service("/rollout/step_return", StepReturn, self.callback_step_return)
        s = rospy.Service("/rollout/start", Trigger, self.callback_start_rollout)
        s = rospy.Service("/guidance/info", GuidanceInfo, self.callback_guidance)
        # Data
        self.consistency = utils.DictToStruct(
            **{
                "experiment": "",
                "initialized": False
            }
        )
        self.robot_state = None
        self.odometry = None
        self.goal = None
        self.guide = None  # Guidance()
        self.is_guided = False
        self.stair = None
        self.debug = []
        # coefficients
        # Distance to the center plane coefficient
        rospy.spin()

    def callback_guidance(self, _):
        self.is_guided = True
        # the last episode data sending
        if self.guide.done:
            self.send_to_backend()
        return GuidanceInfoResponse(
            # epsilon=self.guide.get_epsilon(),
            epsilon=self.guide.get_progress(),
            done=self.guide.done
        )

    def update_goal(self):
        self.goal = self.goal_info.call(GoalInfoRequest())

    def update_stair(self):
        self.stair = self.stair_info.call(StairInfoRequest())

    def callback_start_rollout(self, req):
        self.rollout_state.time_step = 1
        self.update_goal()
        self.update_stair()
        dist = utils.get_distance(self.odometry.pose.pose.position, self.goal)
        self.rollout_state.closest_distance = dist
        self.rollout_state.maximum_distance = dist
        self.rollout_state.started = True
        return TriggerResponse(success=True, message="")

    def check_consistency(self, req):
        """
        :param req:
        :return: bool, need to reset
        """
        if not self.consistency.initialized:
            self.consistency.experiment = req.experiment
            self.consistency.initialized = True
            # the only place where we initialize Guidance()
            if req.use_penalty_angular:
                penalty_type = "angular"
            elif req.use_penalty_deviation:
                penalty_type = "deviation"
            else:
                penalty_type = "free"
            self.guide = Guidance(penalty_type)

            if penalty_type != "free":
                self.guide.set_need_to_penalize(True)
                self.guide.send_log(f"Need to penalize {penalty_type}!")
            else:
                self.guide.send_log(f"No penalties.!")
            return False
        else:
            if self.consistency.experiment != req.experiment:
                return True
            else:
                return False

    def callback_new_rollout(self, req):
        """
        New rollout callback.
        This method check if new experiment was started. If so, then reinitialize all data attributes.
        Then, the consistency is rechecked.
        After the rollout state is initialized.
        :param req:
        :return:
        """
        print("new rollout", req)
        self.debug = []
        need_to_reset = self.check_consistency(req)
        if need_to_reset:
            self.consistency = utils.DictToStruct(
                **{
                    "experiment": "",
                    "initialized": False
                }
            )
            self.robot_state = None
            self.odometry = None
            self.goal = None
            self.is_guided = False
            self.stair = None

        _ = self.check_consistency(req)

        self.rollout_state.reset()
        self.rollout_state.exp_series = rospy.get_param("exp_series_name")
        self.rollout_state.set_fields(req)

        self.guide.set_seq(self.rollout_state.seq)
        return NewRolloutResponse(received=True)

    def callback_step_return(self, _):
        reward = self.rollout_state.step_reward
        self.rollout_state.step_reward = 0.

        # reshape in case of penalties
        reward = self.guide.reshape_reward(reward)
        if "tipping over" in self.rollout_state.accidents:
            reward -= 1.0
        self.rollout_state.episode_reward += reward

        self.rollout_state.time_step += 1
        if self.rollout_state.time_step == self.rollout_state.time_step_limit:
            self.rollout_state.done = True

        if self.rollout_state.done:
            self.send_to_backend()
            if self.is_guided:
                self.guide.update(
                    self.rollout_state.episode_reward,  # episode reward
                    self.rollout_state.progress  # episode progress
                )
        # print("EPISODE REWARD", self.rollout_state.episode_reward)
        return StepReturnResponse(reward=reward, done=self.rollout_state.done)

    def send_to_backend(self):
        log = self.guide.log_string if self.guide.log_update else ""
        self.guide.reset_sync_log()
        angular_mean = np.mean(self.rollout_state.episode_angular) if len(self.rollout_state.episode_angular) > 0 else 0.
        deviation_mean = np.mean(self.rollout_state.episode_deviation) if len(self.rollout_state.episode_deviation) > 0 else 0.
        # coef_mean = np.mean(self.guide.used_penalty) if len(self.guide.used_penalty) > 0 else -0.1
        # if np.isnan(self.rollout_state.episode_reward) or not isinstance(self.rollout_state.episode_reward, float):
        #     self.rollout_state.episode_reward = 0.
        #     debug_value = 0.
        #     log += "\nnan is detected"
        # else:
        #     pass
        debug_value = self.guide.get_epsilon()
        self.rollout_analytics.publish(
            RolloutAnalytics(
                exp_series=self.rollout_state.exp_series,
                experiment=self.rollout_state.experiment,
                seq=self.rollout_state.seq,
                sensors=self.rollout_state.sensors,
                arm=self.rollout_state.arm,
                angular=self.rollout_state.angular,
                progress=self.rollout_state.progress,
                reward=self.rollout_state.episode_reward,
                angular_m=angular_mean,
                deviation=deviation_mean,
                accidents=self.rollout_state.accidents,
                time_steps=self.rollout_state.time_step,
                log=log,
                debug=debug_value
            )
        )

    def callback_robot_state(self, msg):
        self.robot_state = msg

    def callback_safety_deviation(self, msg):
        self.rollout_state.episode_deviation.append(msg.data)
        if self.rollout_state.use_penalty_deviation and msg.data != 0.:
            self.guide.safety_push(msg.data)

    def callback_safety_angular(self, msg):
        self.rollout_state.episode_angular.append(msg.data)
        if self.rollout_state.use_penalty_angular and msg.data != 0.:
            self.guide.safety_push(msg.data)

    def callback_odometry(self, msg):
        self.odometry = msg
        if self.rollout_state.done or not self.rollout_state.started:
            return

        # distance check
        dist = utils.get_distance(self.odometry.pose.pose.position, self.goal)
        if dist < self.rollout_state.closest_distance:
            diff = self.rollout_state.closest_distance - dist
            # ad hoc clipping
            if self.rollout_state.progress < 1.0:
                self.rollout_state.progress += diff / self.rollout_state.maximum_distance
                self.rollout_state.step_reward += diff / self.rollout_state.maximum_distance
        if dist < 0.0:
            self.rollout_state.closest_distance = 0.
            # print("Done 2")
            self.rollout_state.done = True
        elif dist > 1.2 * self.rollout_state.maximum_distance:
            # print("Done 2-5")
            self.rollout_state.closest_distance = dist
            self.rollout_state.done = True
        else:
            self.rollout_state.closest_distance = dist
        # tipping over check
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]
        )
        accident = False
        if roll > np.pi / 2:
            self.rollout_state.accidents = "Front tipping over"
            accident = True
        elif roll < -np.pi / 2:
            self.rollout_state.accidents = "Rear tipping over"
            accident = True
        if pitch > np.pi / 2:
            self.rollout_state.accidents = "Right tipping over"
            accident = True
        elif pitch < -np.pi / 2:
            self.rollout_state.accidents = "Left tipping over"
            accident = True
        if accident:
            self.rollout_state.done = True


if __name__ == '__main__':
    Monitor()
