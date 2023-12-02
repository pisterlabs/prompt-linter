#!/usr/bin/env python


# -*- coding: utf-8 -*-

import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import firefly_env, firefly_multiagent_env, firefly_end2end_env
from gym.envs.registration import register
from geometry_msgs.msg import Point, PointStamped, Pose, PoseWithCovarianceStamped, PoseArray
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
# from openai_ros.msg import RLExperimentInfo
import os
import numpy as np
from std_msgs.msg import Float64, Int16
import tf
from copy import deepcopy

dict_joints ={"Nose":0,
              "LEye":1,
              "REye":2,
              "LEar":3,
              "REar":4,
              "LShoulder":5,
              "RShoulder":6,
              "LElbow":7,
              "RElbow":8,
              "LWrist":9,
              "RWrist":10,
              "LHip":11,
              "RHip":12,
              "LKnee":13,
              "Rknee":14,
              "LAnkle":15,
              "RAnkle":16}
ind_joints = {v:k for k,v in dict_joints.items()}

alpha_to_gt_joints = {"Nose":1,
              "LEye":1,
              "REye":1,
              "LEar":1,
              "REar":1,
              "LShoulder":2,
              "RShoulder":3,
              "LElbow":4,
              "RElbow":5,
              "LWrist":6,
              "RWrist":7,
              "LHip":8,
              "RHip":9,
              "LKnee":10,
              "Rknee":11,
              "LAnkle":12,
              "RAnkle":13}

gt_joints={'actor::actor_pose':0,
             'actor::Head':0,
             'actor::LeftArm':0,
             'actor::RightArm':0,
             'actor::LeftForeArm':0,
             'actor::RightForeArm':0,
             'actor::LeftFingerBase':0,
             'actor::RightFingerBase':0,
             'actor::LeftUpLeg':0,
             'actor::RightUpLeg':0,
             'actor::LeftLeg':0,
             'actor::RightLeg':0,
             'actor::LeftFoot':0,
             'actor::RightFoot':0,
             }


class DeepFollowEnv(firefly_end2end_env.FireflyMultiAgentEnv):
    def __init__(self,**kwargs):
        """
        This Task Env is designed for MAVs to track a human target
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/ros_ws_abspath", None)
        # assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        # assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
        #                                        " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
        #                                        "/src;cd " + ros_ws_abspath + ";catkin_make"

        # ROSLauncher(rospackage_name="turtlebot_gazebo",
        #             launch_file_name="start_world.launch",
        #             ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/firefly/config",
                               yaml_file_name="formation.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(DeepFollowEnv, self).__init__(ros_ws_abspath,**kwargs)

        # Get WorkSpace Cube Dimensions
        self.work_space_max = rospy.get_param("/work_space/max")
        self.work_space_min = rospy.get_param("/work_space/min")

        self.image_width = rospy.get_param("/image_width")
        self.image_height = rospy.get_param("/image_height")

        self.max_distance_to_imgcenter = numpy.linalg.norm(np.array([self.image_width/2,self.image_height/2]))

    # COCO Joints:     {0,  "Nose"}, {1,  "LEye"}, {2,  "REye"}, {3,  "LEar"}, {4,  "REar"}, {5,  "LShoulder"}, {6,  "RShoulder"}, {7,  "LElbow"}, {8,  "RElbow"},
    #                  {9,  "LWrist"}, {10, "RWrist"}, {11, "LHip"}, {12, "RHip"}, {13, "LKnee"}, {14, "Rknee"}, {15, "LAnkle"}, {16, "RAnkle"},

    # Gazebo Actor: 'actor::actor_pose', 'actor::Hips', 'actor::LHipJoint', 'actor::LeftUpLeg', 'actor::LeftLeg', 'actor::LeftFoot', 'actor::LeftToeBase', 'actor::RHipJoint', 'actor::RightUpLeg',
    #               'actor::RightLeg', 'actor::RightFoot', 'actor::RightToeBase', 'actor::LowerBack', 'actor::Spine', 'actor::Spine1', 'actor::Neck', 'actor::Neck1', 'actor::Head', 'actor::LeftShoulder',
    #               'actor::LeftArm', 'actor::LeftForeArm', 'actor::LeftHand', 'actor::LeftFingerBase','actor::LeftHandIndex1', 'actor::LThumb', 'actor::RightShoulder', 'actor::RightArm','actor::RightForeArm',
    #               'actor::RightHand', 'actor::RightFingerBase', 'actor::RightHandIndex1', 'actor::RThumb'



        # observations =  person root in image, person height in image
        low = np.array([\
                         0, 0,\
                        -1,-1,\
                        -1,-1,\
                         0
                        ])
        # low = []
        # joints_low = np.array([0,0])
        # low = np.concatenate((low,np.tile(joints_low,34))) # 17 x,y positions
        # low = np.concatenate((low,np.tile(joints_low,4))) # lhip and rhip x,y positions
        # low = np.append(low,0)
        # low = np.concatenate((low,np.array([self.work_space_min,self.work_space_min,-np.pi,-np.pi]))) #x,y,theta1, theta2
        # prob_low = np.zeros((1,17*2+3)) #17 joints
        # low = np.concatenate((low,prob_low[0]),axis=0)
        # low = prob_low

        high = np.array([\
                        1, 1,\
                        1, 1,\
                        1, 1,\
                        1
                        ])
        # high=[]
        # joints_high = np.array([1,1])
        # high = np.concatenate((high,np.tile(joints_high,34))) # 17 x,y positions
        # high = np.concatenate((high,np.tile(joints_high,4))) # lhip,rhip x,y positions
        # high = np.append(high,2*np.linalg.norm([self.work_space_max,self.work_space_max])) #considering only inter-agent distance
        # high = np.concatenate((high,np.array([self.work_space_max,self.work_space_max,np.pi,np.pi]))) #x,y,theta1, theta2
        # prob_high = np.ones((1,17*2+3)) #17 joints
        # high = np.concatenate((high,prob_high[0]),axis=0)
        # high = prob_high

        # Defining Observation Space
        self.observation_space = spaces.Box(low,high)

        # Defining Action Space
        self.control_x_max = rospy.get_param("/work_space/vx_max")
        self.control_x_min = rospy.get_param("/work_space/vx_min")
        self.control_y_max = rospy.get_param("/work_space/vy_max")
        self.control_y_min = rospy.get_param("/work_space/vy_min")
        self.control_z_max = rospy.get_param("/work_space/vz_max")
        self.control_z_min = rospy.get_param("/work_space/vz_min")

        self.ACTION_TYPE = rospy.get_param("/actions")
        if self.ACTION_TYPE=="discrete":
            self.action_space = spaces.Discrete(7)
        else:
            self.action_space = spaces.Box(np.array([self.control_x_min,self.control_y_min,self.control_z_min]),np.array([self.control_x_max,self.control_y_max,self.control_z_max]))

        # Desired distance to target
        self.desired_distance = rospy.get_param("/desired_distance")

        # Desired height from target
        self.desired_height = rospy.get_param("/desired_height")

        # Prediction Horizon
        self.prediction_horizon = rospy.get_param("/prediction_horizon")

        # Limit on number of times the person is lost
        self.person_lost_threshold = rospy.get_param("/person_lost_threshold")

        # If performance is great then give highest reward
        self.happiness_threshold = rospy.get_param("/happiness_threshold")

        # If performance is great then give highest reward
        self.desired_height_ratio = rospy.get_param("/desired_height_ratio")

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        #Number of steps per episode
        self.step_limit = rospy.get_param("/episode_steps");

        #Number of steps per episode
        self.flow_threshold = rospy.get_param("/flow_threshold");

        # pretraining flag
        self.pretrain = rospy.get_param("/pretrain");

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.surface_reward = rospy.get_param("/surface_reward")
        self.forwards_reward = rospy.get_param("/forwards_reward")
        self.desired_distance_reward = rospy.get_param("/desired_distance_reward")
        self.desired_velocity_reward = rospy.get_param("/desired_velocity_reward")
        self.image_centering_reward = rospy.get_param("/image_centering_reward")
        self.joint_detections_reward_gain = rospy.get_param("/joint_detections_reward_gain")
        self.desired_height_reward = rospy.get_param("/desired_height_reward")

        self.cumulated_steps = 0.0
        machine_name = '/machine_'+str(self.robotID);rotors_machine_name = '/firefly_'+str(self.robotID)
        neighbor_name = '/machine_'+str((self.robotID+1)%self.num_robots);rotors_neighbor_name = '/firefly_'+str((self.robotID+1)%self.num_robots)
        self.reward_topic = machine_name+"/reward"
        self._reward_pub = rospy.Publisher(self.reward_topic, Float64, queue_size=1)
        self.reward_msg = Float64()

        self.triangulated_root_topic = machine_name+"/triangulated_root"
        self.triangulated_pub = rospy.Publisher(self.triangulated_root_topic, PoseArray, queue_size=1)
        self.triangulated_root = PoseArray()

    def _set_init_pose(self):
        """Sets the Robot in its init pose. Sart with null control input
        """
        self.move_base([0,0,-1], self.prediction_horizon,10, init=True)
        return True

    def _check_all_systems_ready(self):
        """ Wait for actor to respawn and move """
        rospy.logwarn("Wait for actor to respawn and move")
        rospy.sleep(5)


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start of an episode :return:
        """

        # For Info Purposes
        self.cumulated_reward = 0.0
        self.cumulated_steps = 0.0
        self.person_lost = 0.0
        self.good_behavior = 0.0
        self.lost_target_flag = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        # Target Position. Definition in Robot Environment (firefly_env)
        # self.desired_target = self.get_gt_target()
        # des_target = Point()
        # des_target.x = self.desired_target.pose.pose.position.x
        # des_target.y = self.desired_target.pose.pose.position.y
        # des_target.z = self.desired_target.pose.pose.position.z

        pose = self.get_gt_odom()
        pos = Point()
        pos.x = pose.position.x
        pos.y = pose.position.y
        pos.z = pose.position.z

        # feedback = self.get_detections_feedback()
        # bb_center = Point()
        # bb_center.x = feedback.xcenter
        # bb_center.y = feedback.ycenter
        # bb_center.z = 0.0
        # self.prev_feedback = feedback
        #
        # img_center = Point()
        # img_center.x = self.image_width/2
        # img_center.y = self.image_height/2
        # img_center.z = 0.0
        # bbox = self.get_alphapose_bbox()
        #
        # person_height = abs(feedback.feet_raw - feedback.head_raw)
        # height_ratio = (person_height/self.image_height)
        self.previous_distance = 100 #some large value
        self.previous_height_ratio = self.desired_height_ratio
        self.prev_center = [0.5,0.5] #initialize to image center
        self.detection_data = 1
        self.half_visible = 0
        self._MAV_out_ws = False

    def _set_action(self, action):
        """
        Provide way-point for low-level controller
        """
        rospy.logdebug("Start Set Action ==>"+str(action))
        if self.ACTION_TYPE == "discrete":
            control = []
            if action == 0: #FORWARD
                control.append(self.control_x_max)
                control.append(0)
                control.append(0)
                self.last_action = "FORWARD"

            elif action == 1: #BACKWARD
                control.append(self.control_x_min)
                control.append(0)
                control.append(0)
                self.last_action = "BACKWARD"

            elif action == 2: #UP
                control.append(0)
                control.append(0)
                control.append(self.control_z_min)
                self.last_action = "UP"

            elif action == 3: #DOWN
                control.append(0)
                control.append(0)
                control.append(self.control_z_max)
                self.last_action = "DOWN"

            elif action == 4: #UP
                control.append(0)
                control.append(self.control_y_min)
                control.append(0)
                self.last_action = "LEFT"

            elif action == 5: #DOWN
                control.append(0)
                control.append(self.control_y_max)
                control.append(0)
                self.last_action = "RIGHT"

            elif action == 6: #DOWN
                control.append(0)
                control.append(0)
                control.append(0)
                self.last_action = "STOP"
        elif self.pretrain:
            action = self.get_latest_command()
            control = []
            control.append(action.velocity.x)
            control.append(action.velocity.y)
            control.append(action.velocity.z)
        else:
            control = action

        # We tell MAV the action to execute
        self.move_base(control, self.prediction_horizon, update_rate=50)


    def _get_obs(self):
        """
        Read robot state
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the pose of the MAV
        pose = self.get_gt_odom()
        pose_neighbor = self.get_neighbor_gt_odom()
        target = self.get_gt_target()
        # target = self.get_target()
        # target_velocity = self.get_target_velocity()
        # feedback = self.get_detections_feedback()
        alphapose = self.get_alphapose()
        alphapose_neighbor = self.get_alphapose_neighbor()
        bbox = self.get_alphapose_bbox()
        bbox_neighbor = self.get_alphapose_bbox_neighbor()
        joint_detections = np.array(alphapose.res).reshape((17,3))
        joint_detections_neighbor = np.array(alphapose_neighbor.res).reshape((17,3))
        bbox_coordinates = np.array(bbox.bbox)
        bbox_coordinates_neighbor = np.array(bbox_neighbor.bbox)
        detection = self.get_alphapose_detection()
        detection_neighbor = self.get_alphapose_detection_neighbor()

        # self.uncertainty_cost([], epsilon = -2, lower_bound = -4)
        # if self.lost_target_flag:
        #     detection = 0
        # else:
        #     detection = 1
        # self.joints = deepcopy(joint_detections)
        # self.joints_neighbor = deepcopy(joint_detections_neighbor)
        # normalizing features
        # joint_detections[:,0] = [float(joint_detections[k,0]/self.image_width) for k in range(len(joint_detections[:,0]))]
        # joint_detections[:,1] = [float(joint_detections[k,1]/self.image_height) for k in range(len(joint_detections[:,1]))]
        # joint_detections_neighbor[:,0] = [float(joint_detections_neighbor[k,0]/self.image_width) for k in range(len(joint_detections_neighbor[:,0]))]
        # joint_detections_neighbor[:,1] = [float(joint_detections_neighbor[k,1]/self.image_height) for k in range(len(joint_detections_neighbor[:,1]))]

        # joints_flattened = joint_detections[:,0:2].reshape(34,)
        # joints_flattened = joint_detections[11:13,0:2].reshape(4,)
        # joints_neighbor_flattened = joint_detections_neighbor[:,0:2].reshape(34,)
        # joints_neighbor_flattened = joint_detections_neighbor[11:13,0:2].reshape(4,)
        # joint_probabilities = joint_detections[:,2]

        # We round to only two decimals to avoid very big Observation space position, velocity of robot, position  and velocity of target,
        # center of image, head and feet position
        if bbox_coordinates[0] <= 2  or bbox_coordinates[1]<=2 or bbox_coordinates[2]>=self.image_width-2 or bbox_coordinates[3]>=self.image_height-2:
            self.half_visible = 1
        else:
            self.half_visible = 0
        xcenter = round((bbox_coordinates[2] - bbox_coordinates[0])/2) + bbox_coordinates[0]
        ycenter = round((bbox_coordinates[3] - bbox_coordinates[1])/2) + bbox_coordinates[1]

        # '''2-D inter-agent distance'''
        # inter_agent_distance = np.linalg.norm([pose.position.x - pose_neighbor.position.x, pose.position.y - pose_neighbor.position.y])
        '''Angle w.r.t actor for Mav and neighbor'''
        # theta1 = np.arctan2(pose.position.y-target.pose.pose.position.y, pose.position.x-target.pose.pose.position.x)
        # theta2 = np.arctan2(pose_neighbor.position.y-target.pose.pose.position.y, pose_neighbor.position.x-target.pose.pose.position.x)

        ''' Absolute Agent yaw'''
        # (trans,rot) = self.listener.lookupTransform( 'world_ENU',self.rotors_machine_name+'/base_link', rospy.Time(0))
        # (r,p,y) = tf.transformations.euler_from_quaternion(rot)
        # theta1 = y
        #
        # (trans,rot) = self.listener.lookupTransform( 'world_ENU',self.rotors_neighbor_name+'/base_link', rospy.Time(0))
        # (r,p,y) = tf.transformations.euler_from_quaternion(rot)
        # theta2 = y

        observations =  [\
                        float(detection.data*xcenter/self.image_width), float(detection.data*ycenter/self.image_height), \
                        float(detection.data*(xcenter/self.image_width - self.prev_center[0])),  float(detection.data*(ycenter/self.image_height - self.prev_center[1])), \
                        float(detection.data*(pose.position.x-target.pose.pose.position.x)/self.work_space_max),\
                        float(detection.data*(pose.position.y-target.pose.pose.position.y)/self.work_space_max),\
                        float(detection.data*(pose.position.z-target.pose.pose.position.z)/self.work_space_max)
                        ]
                        #+list(joints_flattened)+list(joints_neighbor_flattened)+[pose.position.x - pose_neighbor.position.x, pose.position.y - pose_neighbor.position.y, theta1, theta2]

                        #+[inter_agent_distance]

                        # detection.data, \
                        # abs(pose.position.z/15)
        #                 # feedback.xcenter - self.prev_feedback.xcenter, \
        #                 # feedback.ycenter - self.prev_feedback.ycenter \
                        #+ list(joint_probabilities)
        # observations  =  list(joints_flattened) + [detection.data*xcenter/self.image_width, detection.data*ycenter/self.image_height, \
        #                                             detection.data*abs(bbox_coordinates[3]-bbox_coordinates[1])/self.image_height]

        self.detection_data = detection.data
        rospy.logwarn("Center==>"+str(xcenter)+" "+str(ycenter))
        self.prev_center = [observations[0],observations[1]]


        rospy.logwarn("Observations==>"+str(observations))
        rospy.logdebug("Observations==>"+str(observations))
        rospy.logdebug("END Get Observation ==>")
        # self.act_pretrained, _states = self.pretrained_model.predict(observation=observations[0:6], deterministic = True)
        # self.act_pretrained_joints, _states = self.pretrained_model_joints.predict(observation=observations[5:39], deterministic = True)
        return observations


    def _is_done(self, observations):

        if self._episode_done:
            rospy.logerr("Episode DONE==>")
        else:
            # rospy.logerr("MAV is operating in workspace==>"+str(self.cumulated_steps))
            if self.cumulated_steps == self.step_limit:
                rospy.logerr("Episode Step Limit Reached")
                self._episode_done = True

            pose = self.get_gt_odom()

            # We see if we are outside the workspace
            if pose.position.x <= self.work_space_max and pose.position.x > self.work_space_min:
                self._MAV_out_ws = False
                if pose.position.y <= self.work_space_max and pose.position.y > self.work_space_min:
                    self._MAV_out_ws = False
                    if pose.position.z <= -1 and pose.position.z > -15:
                        rospy.logdebug("MAV Position is OK ==>["+str(pose.position.x)+","+str(pose.position.y)+","+str(pose.position.z)+"]")
                        self._MAV_out_ws = False

                        # We see if it got to the desired point
                        # [flag,distance_from_target] = self.is_in_desired_distance(observations,epsilon=0.5)
                        # if distance_from_target > 12:
                        #     rospy.logerr("MAV too Far from the Person")
                        #     self._episode_done = True
                        # if distance_from_target < 2:
                        #     rospy.logerr("MAV too Close to the Person")
                        #     self._episode_done = True
                    else:
                        rospy.logerr("MAV too Far in Z Pos ==>"+str(pose.position.z))
                        self._MAV_out_ws =  True
                else:
                    rospy.logerr("MAV too Far in Y Pos ==>"+str(pose.position.y))
                    self._MAV_out_ws = True
            else:
                rospy.logerr("MAV too Far in X Pos ==>"+str(pose.position.x))
                self._MAV_out_ws = True

            # If robot loses the person stop training
            # if not self.lost_target_flag:
            #     rospy.logerr("Person in Cam Position OK ==>["+str(observations[1])+","+str(observations[2])+"]")
            # else:
            #     self.person_lost+=1

            # if self.person_lost > self.person_lost_threshold:
            #     rospy.logerr("Person lost too many times")
            #     self._episode_done = True

        return self._episode_done

    def _compute_reward(self, observations, done):
        reward = 0
        error = 0
        # height_ratio = 0
        if not done:
            # Cost for uncertainty in tracking and check if target is still visible
            # self.uncertainty_cost(observations, epsilon = -2, lower_bound = -4)
            # Only if  target is in sight and visible
            # if not self.lost_target_flag:
            '''Following reward in world workspace.'''

            # [flag,distance_from_target] = self.is_in_desired_distance(observations,epsilon_up=7,epsilon_down=3.5)
            [flag,distance_from_target] = self.is_in_desired_distance(observations,epsilon=0.5)
            if flag:
                reward += 0.3*self.detection_data
            else:
                reward += 0.01*self.detection_data
            #
            #
            '''Following height reward in world workspace'''
            [flag,height] = self.is_in_MAV_desired_altitude(observations,epsilon=0.5)
            if flag:
                reward += 0.3*self.detection_data
            else:
                reward += 0.01*self.detection_data
            #
            #
            #
            '''Following velocity reward in world workspace'''
            [flag,vel_err] = self.is_in_desired_velocity(observations, epsilon=0.1)
            if flag:
                reward += 0.1*self.detection.data
            else:
                reward+= 0.01*self.detection.data


            '''Perception centering reward'''
            [flag,distance_to_imgcenter] = self.is_image_centered(observations, epsilon=0.1)
            if flag:
                reward += 0.25*self.detection_data
            else:
                reward += 0.01*self.detection_data
            # reward+= -1*(observations[3])*10*distance_to_imgcenter**2 -1*(1-observations[3])*10 + 10
            # reward+= -1*distance_to_imgcenter**2 + 1


            '''Perception height cost / Tracking reward in image space'''
            # [flag,height_ratio] = self.is_desired_height(observations, epsilon=0.05)
            # if flag:
            #     reward += 0.16*self.detection_data*(1-self.half_visible)
            # else:
            #     reward += 0.01*self.detection_data*(1-self.half_visible)

            # reward += 10*detection.data*(-1*(height_ratio-self.desired_height_ratio)**2  + 0.5)
            # if flag:
            #     reward += observations[3]*self.desired_height_reward
            # else:
            #     reward +=0
            # reward += -1*(observations[3])*20*(height_ratio-self.desired_height_ratio)**2 - 1*(1-observations[3])*10 + 10

            '''Minimize person velocity in image plane'''
            # [flag,flow] = self.is_flow_zero(observations, epsilon=0.5)
            # if flag:
            #     reward += 0.2*self.detection.data
            # else:
            #     reward += 0.01*self.detection.data
            #

            # '''Penalize Workspace Violation'''
            # if self._MAV_out_ws:
            #     reward += -0.33
            #
            # ''' Target Safety'''
            # if distance_from_target<1.5:
            #     reward += -0.34
            #
            # '''Penalize Height Violation'''
            # if height > 15:
            #     reward += -0.33


            ''' Reward for Decrease in Distance'''
            # If there has been a decrease in the distance towards target, we reward it
            [flag,rad_distance_from_target] = self.is_in_desired_rad_distance(observations,epsilon_up=7,epsilon_down=3.5)
            delta_error =  rad_distance_from_target - self.previous_distance
            self.previous_distance = rad_distance_from_target
            if delta_error < 0.0 and rad_distance_from_target > np.sqrt(self.desired_distance**2+self.desired_distance**2)+0.5:
                reward += 0.05*self.detection_data
                rospy.logwarn("################### DECREASING ERROR: GOOD JOB ###########################")
            else:
                reward += 0.0
            #
            '''Person joints visiblity'''
            # [flag,joint_vis] = self.joint_detections_reward(observations, epsilon=4)
            # if flag and rad_distance_from_target <= np.sqrt(self.desired_distance**2+self.desired_distance**2)+1.5 and rad_distance_from_target > 6:
            #     reward_new += 1
            # else:
            #     reward_new += 0.01*self.detection.data

            '''Multi-Agent Obstacle Avoidance'''
            # [flag,ia_distance] = self.is_in_desired_multiagent_distance(observations,epsilon=3)
            # if flag and ia_distance<15:
            #     reward += 0
            #     rospy.logwarn("################### NO COLLISION: GOOD JOB ###########################")
            # else:
            #     reward += -0.2


            '''Least Squares Triangulation Error'''
            # [flag,error] = self.get_lsq_triangulation_error(observations,epsilon=50)#epsilon =30 for full body
            # reward += -error

            # target = self.get_gt_target()
            # rospy.logerr("@@@@ 1. d2img_center: " + str(distance_to_imgcenter))
            # rospy.logerr("@@@@ 2. MAV altitude : " + str(height))
            # rospy.logerr("@@@@ 3. flow: " + str(flow))
            # rospy.logerr("@@@@ 3. h_ratio: " + str(height_ratio)+" image_height: "+str(self.image_height))
            # rospy.logerr("@@@@ 4. d2t: " + str(distance_from_target))
            # rospy.logerr("@@@@ 5. velocity err: " + str(vel_err))
            # rospy.logerr("@@@ 7. joint spread: "+ str(joint_vis))
            # rospy.logerr("@@@@ 4. joint_reward: " + str(self.joint_detections_reward_gain*joint_r))
            # rospy.logerr("@@@@ 7. sum(joint_probabilities)/17: " + str(sum(observations[10:])/17))
            # rospy.logerr("@@@@ 8. detection: " + str(observations[3]))
            # rospy.logerr("@@@@ normalized and bounded triangulation error: " + str(error))
            # rospy.logerr("@@@@ inter-agent distance: " + str(ia_distance))
            # rospy.logerr("@@@@ theta1, theta2 " + str(observations[-2])+str(observations[-1]))
            '''
                    else:
                        if self.is_in_desired_position():
                            reward += self.surface_reward
                        else:distance_to_imgcenter/self.max_distance_to_imgcenter
                            reward += -1*self.surface_reward

            '''
#        else:
#            if self.cumulated_steps != self.step_limit:
#                reward+=-20*(self.step_limit - self.cumulated_steps)


        if reward == 1:
                rospy.logerr("")
                rospy.logerr("@@@@@@@@@@@@@@@@@@@@@@@@ AWESOME AGENT @@@@@@@@@@@@@@@@@@@@")
                rospy.logerr("")
        rospy.logwarn("reward=" + str(reward))
        rospy.logdebug("reward=" + str(reward))
        self.reward_msg.data = reward
        self._reward_pub.publish(self.reward_msg)
        self.cumulated_reward += reward
        rospy.logwarn("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logwarn("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    def uncertainty_cost(self,observations, epsilon = -2, lower_bound = -4):
        target = self.get_target()
        reward = -1*(target.pose.covariance[0]+target.pose.covariance[7]+target.pose.covariance[14])
        if reward >= epsilon:
            self.good_behavior+=1
            rospy.logwarn("################### LOW TRACKING UNCERTAINTY: GOOD JOB ###########################")
            self.lost_target_flag = False
        if reward < lower_bound:
            self.lost_target_flag = True
            reward = lower_bound
        return reward

    def is_in_desired_rad_distance(self,observations,  epsilon_up=7, epsilon_down=3):
        """
        It returns True if the current position is similar to the desired position
        """
        target = self.get_gt_target()
        pose = self.get_gt_odom()
        is_in_desired_pos = False
        current_position = Point()
        current_position.x = target.pose.pose.position.x - pose.position.x
        current_position.y = target.pose.pose.position.y - pose.position.y
        current_position.z = target.pose.pose.position.z - pose.position.z

        null_point = Point();null_point.x = 0;null_point.y = 0;null_point.z = 0;

        distance_from_target = self.get_distance_from_point(null_point,current_position)
        if distance_from_target <=epsilon_up and distance_from_target > epsilon_down:
            is_in_desired_pos = True
            self.good_behavior+=1
            rospy.logwarn("################### FOLLOWING: GOOD JOB ###########################")

        return [is_in_desired_pos,distance_from_target]


    def is_in_desired_distance(self,observations,epsilon=0.5):
        """
        It returns True if the current position is similar to the desired position
        """
        target = self.get_gt_target()
        pose = self.get_gt_odom()
        is_in_desired_pos = False
        current_position = Point()
        current_position.x = target.pose.pose.position.x - pose.position.x
        current_position.y = target.pose.pose.position.y - pose.position.y
        current_position.z = 0

        null_point = Point();null_point.x = 0;null_point.y = 0;null_point.z = 0;

        distance_from_target = self.get_distance_from_point(null_point,current_position)
        # if distance_from_target <=epsilon_up and distance_from_target > epsilon_down:
        if abs(distance_from_target-self.desired_distance) <= epsilon:
            is_in_desired_pos = True
            self.good_behavior+=1
            rospy.logwarn("################### FOLLOWING: GOOD JOB ###########################")

        return [is_in_desired_pos,distance_from_target]

    def is_in_MAV_desired_altitude(self,observations,epsilon=0.5):
        """
        It returns True if the current position is above a threshold height
        """
        pose = self.get_gt_odom()
        target = self.get_gt_target()
        height = abs(pose.position.z-target.pose.pose.position.z)
        is_in_desired_pos = False
        if abs(height-self.desired_distance) <= epsilon:
            is_in_desired_pos = True
            self.good_behavior+=1
            rospy.logwarn("################### ALTITUTDE: GOOD JOB ###########################")

        return [is_in_desired_pos, height]



    def is_in_desired_velocity(self,observations, epsilon=0.5):
        """
        It returns True if the current velocity is similar to the desired velocity
        """
        is_in_desired_vel = False
        vel = self.get_velocity()
        target = self.get_gt_target()
        current_vel = Point()
        current_vel.x = vel.twist.twist.linear.y - target.twist.twist.linear.x
        current_vel.y = vel.twist.twist.linear.x - target.twist.twist.linear.y
        current_vel.z = 0#((-vel.twist.twist.linear.z) - target_vel.twist.linear.z)

        null_point = Point();null_point.x = 0;null_point.y = 0;null_point.z = 0;
        err = self.get_distance_from_point(current_vel,null_point)
        if err<epsilon:
            is_in_desired_vel = True
            self.good_behavior+=1
            rospy.logwarn("################### VELOCITY: GOOD JOB ###########################")
        return [is_in_desired_vel, err]

    def is_image_centered(self,observations, epsilon=0.3):
        """
        It returns True if the current position is similar to the desired position
        """
        is_in_desired_pos = False
        bb_center = Point()
        bb_center.x = observations[0]
        bb_center.y = observations[1]
        bb_center.z = 0.0

        img_center = Point()
        img_center.x = 0.5
        img_center.y = 0.5
        img_center.z = 0.0

        distance_to_imgcenter = self.get_distance_from_point(bb_center,img_center)

        if distance_to_imgcenter<= epsilon:
            is_in_desired_pos = True
            self.good_behavior+=1
            rospy.logwarn("################### PERCEIVING: GOOD JOB ###########################")

        return [is_in_desired_pos,distance_to_imgcenter]



    def is_desired_height(self,observations,epsilon=0.03):
        """
        It returns True if the tracking person height ratio w.r.t image height is close to a desired value
        """

        is_in_desired_pos = False
        height_ratio = observations[2]
        if abs(height_ratio-self.desired_height_ratio) <= epsilon:
            is_in_desired_pos = True
            self.good_behavior+=1
            rospy.logwarn("################### HEIGHT_RATIO: GOOD JOB ###########################")
        return [is_in_desired_pos,height_ratio]


    def is_flow_zero(self,observations, epsilon=0.1):
        """
        It returns True if the current position is similar to the desired position
        """
        is_in_desired_pos = False
        flow = Point()
        flow.x = observations[3]
        flow.y = observations[4]
        flow.z = 0.0

        null_point = Point();null_point.x = 0;null_point.y = 0;null_point.z = 0;

        person_flow = self.get_distance_from_point(flow,null_point)

        if person_flow < epsilon:
            is_in_desired_pos = True
            self.good_behavior+=1
            rospy.logwarn("################### FLOW: GOOD JOB ###########################")

        return [is_in_desired_pos,person_flow]

    def joint_detections_reward(self,observations, epsilon=0.5):
        # sum_obs = sum(observations[0:34])/34 #17 human joints
        obsx = [];obsy=[]

        sum_obsx = 0;sum_obsy=0
        std_obsx = 0;std_obsy = 0;
        for k in range(0,34,2):
            sum_obsx+=observations[k]
            sum_obsy+=observations[k+1]
        mean_obsx = sum_obsx/17;mean_obsy = sum_obsy/17;
        for k in range(0,34,2):
            std_obsx += abs(observations[k]-mean_obsx)
            std_obsy += abs(observations[k+1]-mean_obsy)
        sum_obs  = std_obsx+std_obsy
        is_in_desired_pos = False
        # sum_obs = sum(abs(np.array(observations[0:34]).reshape(17,2)-np.array(observations[0:2])))
        if sum_obs >=epsilon:
            self.good_behavior+=1
            is_in_desired_pos = True
            rospy.logwarn("################### JOINTS: GOOD JOB ###########################")

        return [is_in_desired_pos,sum_obs]


    def is_in_desired_position(self):
        """
        It return True if the tracking of the person is consistent for a given number of time steps
        """
        is_in_desired_pos = False

        if self.good_behavior>=self.happiness_threshold:
            rospy.logwarn("################### HAPPY AGENT ###########################")
            is_in_desired_pos = True

        return is_in_desired_pos


    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def is_in_desired_multiagent_distance(self,observations,epsilon=2):
        """
        It returns True if the current position is similar to the desired position
        """
        is_in_desired_pos = False
        relative_pos = Point()
        relative_pos.x = observations[-4]
        relative_pos.y = observations[-3]
        relative_pos.z = 0.0

        null_point = Point()
        null_point.x = 0.0
        null_point.y = 0.0
        null_point.z = 0.0

        ia_distance = self.get_distance_from_point(relative_pos,null_point)

        if ia_distance >= epsilon :
            is_in_desired_pos = True
        return [is_in_desired_pos,ia_distance]


    def get_lsq_triangulation_error(self,observation,epsilon=0.5):
        # Get the camera intrinsics of both cameras
        P1 = self.get_cam_intrinsic()
        P2 = self.get_neighbor_cam_intrinsic()

        # Convert world coordinates to local camera coordinates for both cameras
        # We get the camera extrinsics as follows
        trans,rot = self.listener.lookupTransform('firefly_1/xtion_rgb_optical_frame','world', rospy.Time(0)) #target to source frame
        (r,p,y) = tf.transformations.euler_from_quaternion(rot)
        H1 = tf.transformations.euler_matrix(r,p,y,axes='sxyz')
        H1[0:3,3] = trans
        trans,rot = self.listener.lookupTransform('firefly_2/xtion_rgb_optical_frame','world', rospy.Time(0)) #target to source frame
        (r,p,y) = tf.transformations.euler_from_quaternion(rot)
        H2 = tf.transformations.euler_matrix(r,p,y,axes='sxyz')
        H2[0:3,3] = trans


        #Concatenate Intrinsic Matrices
        intrinsic = np.array((P1[0:3,0:3],P2[0:3,0:3]))
        #Concatenate Extrinsic Matrices
        extrinsic = np.array((H1,H2))
        joints_gt = self.get_joints_gt()
        error = 0

        self.triangulated_root = PoseArray()
        for k,v in dict_joints.items():

            # Find the mid-point of left and right hip measurements
            p2d1 = self.joints[dict_joints[k],0:2]
            p2d2 = self.joints_neighbor[dict_joints[k],0:2]
            points_2d = np.array((p2d1,p2d2))

            # Solve system of equations using least squares to estimate person position in robot 1 camera frame
            estimate_root  = self.lstsq_triangulation(intrinsic,extrinsic,points_2d,2)
            es_root_cam = Point();es_root_cam.x = estimate_root[0];es_root_cam.y = estimate_root[1];es_root_cam.z = estimate_root[2]
            p = Pose()
            p.position = es_root_cam
            self.triangulated_root.poses.append(p)
            gt = joints_gt.poses[alpha_to_gt_joints[k]].position
            if v>4: #because head, eyes and ears lead to really high error for the actor
            # if v == 11 or v == 12:
                error += self.get_distance_from_point(gt,es_root_cam)

        # Publish all estimates
        self.triangulated_root.header.stamp = rospy.Time()
        self.triangulated_root.header.frame_id = 'world'
        self.triangulated_pub.publish(self.triangulated_root)

        is_in_desired_pos = False
        if error<=epsilon:
            error = 0.4*error/epsilon
            is_in_desired_pos = True
        else:
            error = 0.4

        return [is_in_desired_pos,error]


    def lstsq_triangulation(self,intrinsic,extrinsic,points_2d,cams):
        '''
        Least Squares Triangulation
        '''

        a=[]
        b=[]
        norm_points = []
        w = []

        for cam in range(cams):
            extr = extrinsic[cam,:3,:]
            # convert to homogeneous coordinates
            point = np.append(points_2d[cam,:], 1)

            # normalize the 2D points using the intrinsic parameters
            norm_points.append(np.matmul(np.linalg.inv(intrinsic[cam,:,:]), point))
            # we'll use equation 14.42 of the CV book by Simon Prince.
            # form of the equation is Ax=b.

            # generate the matrix A and vector b
            a.append(np.outer((norm_points[-1])[0:2], extr[2, 0:-1]) - extr[0:2, 0:-1])
            b.append(extr[0:-1, -1] - extr[-1, -1] * (norm_points[-1])[0:2])
            # print((norm_points[-1])[0:2])

        A = np.concatenate(a)
        B = np.concatenate(b)

        # solve with least square estimate
        x = np.linalg.lstsq(A, B, rcond=None)[0]

        return x
