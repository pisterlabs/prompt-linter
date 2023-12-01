#!/usr/bin/env python


# -*- coding: utf-8 -*-

import numpy as np
import rospy
import time
import os
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64, Int16
from sensor_msgs.msg import JointState
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PointStamped
from geometry_msgs.msg import PoseArray, PoseStamped, TwistStamped, PoseWithCovarianceStamped, TwistWithCovarianceStamped
from openai_ros.openai_ros_common import ROSLauncher
from uav_msgs.msg import uav_pose
from neural_network_detector.msg import NeuralNetworkFeedback
from alphapose_node.msg import AlphaRes
import message_filters

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

class FireflyMultiAgentGTEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for robot environment.
    """

    def __init__(self,**kwargs):
        """
        Initializes a new FireflyEnv environment.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that the stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controllers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered usefull for AI learning.

        Sensor Topic List:
        * /pose : Reads the estimated pose of the MAV. Type: uav_msgs
        * /target_tracker/pose : Reads the fused estimate of the target person. Type: geometry_msgs

        Actuators Topic List:
        * /command, publishes the desired waypoints and velocity for the robot. Type: uav_msgs

        Args:
        """
        rospy.logdebug("Start FireflyEnv INIT...")
        rospy.logwarn('Setting Up Firefly ENV')
        # Variables that we give through the constructor.
        # None in this case


        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(FireflyMultiAgentGTEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD",**kwargs)

        machine_name = '/machine_'+str(self.robotID);rotors_machine_name = '/firefly_'+str(self.robotID)
        self.rotors_machine_name = rotors_machine_name
        self.num_robots=2
        if self.robotID==self.num_robots:
            neighbor_name = '/machine_'+str((self.robotID+1)%self.num_robots);rotors_neighbor_name = '/firefly_'+str((self.robotID+1)%self.num_robots)
        else:
            neighbor_name = '/machine_'+str(self.robotID+1);rotors_neighbor_name = '/firefly_'+str(self.robotID+1)
        self.rotors_neighbor_name = rotors_neighbor_name

        self.pose_topic = machine_name+"/pose"

        self.velocity_topic = rotors_machine_name+"/ground_truth/odometry"

        self.velocity_neighbor_topic = rotors_neighbor_name+"/ground_truth/odometry"

        self.gt_pose_topic = machine_name+"/pose/groundtruth"

        self.gt_neighbor_pose_topic = neighbor_name+"/pose/groundtruth"

        self.firefly_pose_topic = rotors_machine_name+"/ground_truth/pose_with_covariance"

        self.target_topic = machine_name+"/target_tracker/pose"

        self.gt_target_topic = "/actorpose"

        self.gt_target_vel_topic = "/actorvel"

        self.target_velocity_topic = machine_name+"/target_tracker/twist"

        self.command_topic = machine_name+"/command"

        self.neighbor_command_topic = neighbor_name+"/command"

        self.destination_topic = machine_name+"/destination"

        self.detections_feedback_topic = machine_name+"/object_detections/feedback"

        self.noisy_joints_topic = machine_name+"/noisy_joints"

        self.noisy_joints_neighbor_topic = neighbor_name+"/noisy_joints"

        self.noisy_bbox_topic = machine_name+"/noisy_bbox"

        self.noisy_bbox_neighbor_topic = neighbor_name+"/noisy_bbox"

        self.noisy_detection = machine_name+"/noisy_detection"

        self.noisy_neighbor_detection = neighbor_name+"/noisy_detection"

        self.alphapose_topic = machine_name+"/result_alpha"

        self.alphapose_neighbor_topic = neighbor_name+"/result_alpha"

        self.alphapose_bbox_topic = machine_name+"/result_bbox"

        self.alphapose_bbox_neighbor_topic = neighbor_name+"/result_bbox"

        self.alphapose_detection = machine_name+"/detection"

        self.alphapose_neighbor_detection = neighbor_name+"/detection"

        self.mhmr_topic = "/multihmr_joints"

        self.camera_info = rotors_machine_name+rotors_machine_name+"/xtion/rgb/camera_info"

        self.camera_info_neighbor = rotors_neighbor_name+rotors_neighbor_name+"/xtion/rgb/camera_info"

        self.joints_gt = '/gt_joints'

        self.mpc_command = machine_name+"/command_MPC"



        import tf
        self.listener = tf.TransformListener()




        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()



        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber(self.firefly_pose_topic,PoseWithCovarianceStamped , self._firefly_pose_callback)
        rospy.Subscriber(self.pose_topic, uav_pose, self._pose_callback)

        velocity = message_filters.Subscriber(self.velocity_topic, Odometry)
        self.velocity_cache = message_filters.Cache(velocity,300)

        velocity_neighbor = message_filters.Subscriber(self.velocity_neighbor_topic, Odometry)
        self.velocity_neighbor_cache = message_filters.Cache(velocity_neighbor,300)

        rospy.Subscriber(self.gt_pose_topic, uav_pose, self._gt_pose_callback)

        rospy.Subscriber(self.gt_neighbor_pose_topic, uav_pose,self._gt_neighbor_pose_callback)
        # self.gt_neighbor_cache = message_filters.Cache(gt_neighbor,300);


        rospy.Subscriber(self.target_topic, PoseWithCovarianceStamped, self._target_callback)
        gt_target = message_filters.Subscriber(self.gt_target_topic, Odometry)
        self.gt_target_cache = message_filters.Cache(gt_target,300)


        rospy.Subscriber(self.target_velocity_topic, TwistWithCovarianceStamped, self._target_vel_callback)
        rospy.Subscriber(self.detections_feedback_topic, NeuralNetworkFeedback, self._detections_feedback_callback)


        noisy_joints = message_filters.Subscriber(self.noisy_joints_topic, AlphaRes)
        self.noisy_joints_cache = message_filters.Cache(noisy_joints, 300);self.noisy_joints_cache.registerCallback(self._noisy_joints_callback)

        noisy_joints_neighbor = message_filters.Subscriber(self.noisy_joints_neighbor_topic, AlphaRes)
        self.noisy_joints_neighbor_cache = message_filters.Cache(noisy_joints_neighbor, 300);self.noisy_joints_neighbor_cache.registerCallback(self._noisy_joints_neighbor_callback)

        noisy_bbox = message_filters.Subscriber(self.noisy_bbox_topic, AlphaRes)
        self.noisy_bbox_cache = message_filters.Cache(noisy_bbox, 300);self.noisy_bbox_cache.registerCallback(self._noisy_bbox_callback)

        noisy_bbox_neighbor = message_filters.Subscriber(self.noisy_bbox_neighbor_topic, AlphaRes)
        self.noisy_bbox_neighbor_cache = message_filters.Cache(noisy_bbox_neighbor, 300);self.noisy_bbox_neighbor_cache.registerCallback(self._noisy_bbox_neighbor_callback)

        rospy.Subscriber(self.noisy_detection, Int16,self._noisy_detection_callback)
        rospy.Subscriber(self.noisy_neighbor_detection, Int16,self._noisy_neighbor_detection_callback)


        alphapose = message_filters.Subscriber(self.alphapose_topic, AlphaRes)
        self.alphapose_cache = message_filters.Cache(alphapose,300);self.alphapose_cache.registerCallback(self._alphapose_callback)

        alphapose_neighbor = message_filters.Subscriber(self.alphapose_neighbor_topic, AlphaRes)
        self.alphapose_neighbor_cache= message_filters.Cache(alphapose_neighbor,100);self.alphapose_neighbor_cache.registerCallback(self._alphapose_neighbor_callback)

        alphapose_bbox = message_filters.Subscriber(self.alphapose_bbox_topic, AlphaRes)
        self.alphapose_bbox_cache= message_filters.Cache(alphapose_bbox,300)

        alphapose_bbox_neighbor = message_filters.Subscriber(self.alphapose_bbox_neighbor_topic, AlphaRes)
        self.alphapose_bbox_neighbor_cache = message_filters.Cache(alphapose_bbox_neighbor,300)

        mhmr = message_filters.Subscriber(self.mhmr_topic, PoseArray)
        self.mhmr_cache = message_filters.Cache(mhmr,300)


        rospy.Subscriber(self.alphapose_detection, Int16, self._alphapose_detection_callback)
        rospy.Subscriber(self.alphapose_neighbor_detection, Int16, self._alphapose_neighbor_detection_callback)
        rospy.Subscriber(self.command_topic, uav_pose, self._command_callback)
        rospy.Subscriber(self.camera_info, CameraInfo, self._camera_info_callback)
        rospy.Subscriber(self.camera_info_neighbor, CameraInfo, self._camera_info_neighbor_callback)
        rospy.Subscriber(self.joints_gt, PoseArray, self._joints_gt_callback)
        rospy.Subscriber(self.mpc_command, uav_pose, self._mpc_command_callback)

        self._dest_vel_pub = rospy.Publisher(self.destination_topic, PoseStamped, queue_size=1)
        self._cmd_vel_pub = rospy.Publisher(self.command_topic, uav_pose, queue_size=1)
        self._cmd_neighbor_vel_pub = rospy.Publisher(self.neighbor_command_topic, uav_pose, queue_size=1)
        self._target_init_pub = rospy.Publisher(self.target_topic, PoseWithCovarianceStamped, queue_size=1)



        self.create_circle(radius=8)

        # self.gazebo.pauseSim()

        outPose = uav_pose()
        outPose.header.stamp = rospy.Time.now()
        outPose.header.frame_id="world"

        outPose.POI.x = 0
        outPose.POI.y = 0
        outPose.POI.z = 0

        r = 8#
        t = np.random.choice(63,1);
        outPose.position.x = r*np.cos(self.theta[t[0]])
        outPose.position.y = r*np.sin(self.theta[t[0]])
        outPose.position.z = -r
        self._cmd_vel_pub.publish(outPose)
        rospy.logwarn("Finished FireflyEnv INIT...")

        self._check_all_sensors_ready()
        rospy.logwarn("SENSORS OK!.")
        self._check_publishers_connection()
        rospy.logwarn("PUBLISHERS OK!.")


    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _callback(self,noisy_joints,noisy_joints_neighbor):
        self.noisy_joints = noisy_joints
        self.noisy_joints_neighbor = noisy_joints_neighbor

    def create_circle(self, radius=5):
        self.theta = [k for k in np.arange(0,6.28,0.1)]
        x = radius*np.cos(self.theta)
        y = radius*np.sin(self.theta)
        self.init_circle = [x,y]

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL MODULES READY")
        # self._check_odom_ready()
        # self._check_target_ready()
        self._check_gt_target_ready()
        # self._check_target_vel_ready()
        # self._check_feedback_ready()
        # self._check_alphapose_ready()
        # self._check_neighbor_alphapose_ready()
        #self._check_mhmr_ready()
        self._check_neighbor_ready()
        #self._check_camera_rgb_image_info_ready()
        #self._check_camera_neighbor_rgb_image_info_ready()
        # self._check_noisy_joints()
        # self._check_noisy_joints_neighbor()
        #self._check_bbox_ready()
        #self._check_neighbor_bbox_ready()

        rospy.logdebug("ALL MODULES READY")

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /pose to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            # try:
                self.odom = rospy.wait_for_message(self.pose_topic, uav_pose, timeout=5.0)
                rospy.logdebug("Current /pose READY=>")

            # except:
                # rospy.logerr("Current /pose not ready yet, retrying for getting odom")
        return self.odom

    def _check_odom_ready(self):
        self.firefly_odom = None
        rospy.logdebug("Waiting for /fireflypose to be READY...")
        while self.firefly_odom is None and not rospy.is_shutdown():
            # try:
                self.firefly_odom = rospy.wait_for_message(self.firefly_pose_topic, PoseWithCovarianceStamped, timeout=5.0)
                rospy.logdebug("Current /fireflypose READY=>")

            # except:
                # rospy.logerr("Current /pose not ready yet, retrying for getting odom")
        return self.firefly_odom

    def _check_target_ready(self):
        self.target = None
        rospy.logdebug("Waiting for /target to be READY...")
        while self.target is None and not rospy.is_shutdown():
            try:
                self.target = rospy.wait_for_message(self.target_topic, PoseWithCovarianceStamped, timeout=5.0)
                rospy.logdebug("Current /target READY=>")

            except:
                rospy.logerr("Current /target not ready yet, retrying for getting target")

        return self.target

    def _check_gt_target_ready(self):
        self.gt_target = None
        rospy.logdebug("Waiting for /target to be READY...")
        while self.gt_target is None and not rospy.is_shutdown():
            try:
                self.gt_target = rospy.wait_for_message(self.gt_target_topic, Odometry, timeout=5.0)
                rospy.logdebug("Current /target READY=>")

            except:
                rospy.logerr("Current /target not ready yet, retrying for getting target")

        return self.gt_target

    def _check_target_vel_ready(self):
        self.target_velocity = None
        rospy.logdebug("Waiting for /target to be READY...")
        while self.target_velocity is None and not rospy.is_shutdown():
            try:
                self.target_velocity = rospy.wait_for_message(self.target_velocity_topic, TwistWithCovarianceStamped, timeout=5.0)
                rospy.logdebug("Current /target READY=>")

            except:
                rospy.logerr("Current /target not ready yet, retrying for getting target")

        return self.target_velocity


    def _check_noisy_joints(self):
        self.noisy_joints = None
        rospy.logdebug("Waiting for /object_detections/feedback to be READY...")
        while self.noisy_joints is None and not rospy.is_shutdown():
            try:
                self.noisy_joints = rospy.wait_for_message(self.noisy_joints, AlphaRes, timeout=5.0)
                rospy.logdebug("Current /noisy_gt_joints READY=>")

            except:
                rospy.logerr("Current /noisy_gt_joints not ready yet in env"+str(self.env_id)+", retrying for getting target")

        return self.noisy_joints

    def _check_noisy_joints_neighbor(self):
        self.noisy_joints_neighbor = None
        rospy.logdebug("Waiting for /noisy_gt_joints to be READY...")
        while self.noisy_joints_neighbor is None and not rospy.is_shutdown():
            try:
                self.noisy_joints_neighbor = rospy.wait_for_message(self.noisy_joints_neighbor, AlphaRes, timeout=5.0)
                rospy.logdebug("Current /noisy_gt_joints READY=>")

            except:
                rospy.logerr("Current /noisy_gt_joints neighbor not ready yet in env"+str(self.env_id)+", retrying for getting target")

        return self.noisy_joints_neighbor


    def _check_feedback_ready(self):
        self.feedback = None
        rospy.logdebug("Waiting for /object_detections/feedback to be READY...")
        while self.feedback is None and not rospy.is_shutdown():
            try:
                self.feedback = rospy.wait_for_message(self.detections_feedback_topic, NeuralNetworkFeedback, timeout=5.0)
                rospy.logdebug("Current /object_detections/feedback READY=>")

            except:
                rospy.logerr("Current /object_detections/feedback not ready yet, retrying for getting target")

        return self.feedback

    def _check_mhmr_ready(self):
        self.mhmr = None
        rospy.logdebug("Waiting for /alphapose to be READY...")
        while self.mhmr is None and not rospy.is_shutdown():
            try:
                self.mhmr = rospy.wait_for_message(self.mhmr_topic, PoseArray, timeout=5.0)
                rospy.logdebug("Current /multihmr READY=>")

            except:
                rospy.logerr("Current "+self.mhmr_topic+" not ready yet in env"+str(self.env_id)+", retrying for getting target")

        return self.mhmr

    def _check_alphapose_ready(self):
        self.alphapose = None
        rospy.logdebug("Waiting for /alphapose to be READY...")
        while self.alphapose is None and not rospy.is_shutdown():
            try:
                self.alphapose = rospy.wait_for_message(self.alphapose_topic, AlphaRes, timeout=5.0)
                rospy.logdebug("Current /alphapose READY=>")

            except:
                rospy.logerr("Current "+self.alphapose_topic+" not ready yet in env"+str(self.env_id)+", retrying for getting target")

        return self.alphapose

    def _check_neighbor_alphapose_ready(self):
        self.alphapose_neighbor = None
        rospy.logdebug("Waiting for neighbor /alphapose to be READY...")
        while self.alphapose_neighbor is None and not rospy.is_shutdown():
            try:
                self.alphapose_neigbor = rospy.wait_for_message(self.alphapose_neighbor_topic, AlphaRes, timeout=5.0)
                rospy.logdebug("Current /alphapose READY=>")

            except:
                rospy.logerr("Current neighbor /alphapose not ready yet in env"+str(self.env_id)+", retrying for getting target")

        return self.alphapose_neighbor

    def _check_bbox_ready(self):
        self.bbox = None
        rospy.logdebug("Waiting for neighbor /alphapose to be READY...")
        while self.bbox is None and not rospy.is_shutdown():
            try:
                self.bbox = rospy.wait_for_message(self.alphapose_bbox_topic, AlphaRes, timeout=30.0)
                rospy.logdebug("Current /alphapose neighbor READY=>")
            except:
                rospy.logerr("Current "+self.alphapose_bbox_topic+" not ready yet for env:"+str(self.env_id)+", retrying for getting target")
                self.reset()
        return self.bbox

    def _check_neighbor_bbox_ready(self):
        self.bbox_neighbor = None
        rospy.logdebug("Waiting for neighbor /alphapose to be READY...")
        while self.bbox_neighbor is None and not rospy.is_shutdown():
            try:
                self.bbox_neighbor = rospy.wait_for_message(self.alphapose_bbox_neighbor_topic, AlphaRes, timeout=30.0)
                rospy.logdebug("Current /alphapose neighbor READY=>")
            except:
                rospy.logerr("Current neighbor "+self.alphapose_bbox_neighbor_topic+ " not ready yet for env:"+str(self.env_id)+", retrying for getting target")
                self.reset()
        return self.bbox_neighbor

    def _check_neighbor_ready(self):
        self.gt_neighbor_odom = None
        while self.gt_neighbor_odom is None and not rospy.is_shutdown():
            try:
                self.gt_neighbor_odom = rospy.wait_for_message(self.gt_neighbor_pose_topic, uav_pose, timeout=100.0)
                rospy.logdebug("Current "+ self.gt_neighbor_pose_topic+"/pose READY=>")
            except:
                rospy.logerr("Current "+self.gt_neighbor_pose_topic+" not ready yet, retrying for getting target")
        return self.gt_neighbor_odom

    def _check_camera_depth_image_raw_ready(self):
        self.camera_depth_image_raw = None
        rospy.logdebug("Waiting for /camera/depth/image_raw to be READY...")
        while self.camera_depth_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_depth_image_raw = rospy.wait_for_message("/camera/depth/image_raw", Image, timeout=5.0)
                rospy.logdebug("Current /camera/depth/image_raw READY=>")

            except:
                rospy.logerr("Current /camera/depth/image_raw not ready yet, retrying for getting camera_depth_image_raw")
        return self.camera_depth_image_raw


    def _check_camera_rgb_image_info_ready(self):
        self.P1 = None
        rospy.logdebug("Waiting for camera info to be READY...")
        while self.P1 is None and not rospy.is_shutdown():
            try:
                self.caminfo = rospy.wait_for_message(self.camera_info, CameraInfo, timeout=10.0)
                self.P1 = np.array([self.caminfo.P[0:4],self.caminfo.P[4:8],self.caminfo.P[8:12]])
                rospy.logdebug("Current camera info READY=>")

            except:
                rospy.logerr("Current camera info not ready yet")
        return self.P1

    def _check_camera_neighbor_rgb_image_info_ready(self):
        self.P2 = None
        rospy.logdebug("Waiting for neighbor camera info to be READY...")
        while self.P2 is None and not rospy.is_shutdown():
            try:
                self.caminfo_neighbor = rospy.wait_for_message(self.camera_info_neighbor, CameraInfo, timeout=10.0)
                self.P2 = np.array([self.caminfo_neighbor.P[0:4],self.caminfo_neighbor.P[4:8],self.caminfo_neighbor.P[8:12]])
                rospy.logdebug("Current neighbor camera info READY=>")

            except:
                rospy.logerr("Current neighbor camera info not ready yet")
        return self.P2

    def _check_camera_rgb_image_raw_ready(self):
        self.camera_rgb_image_raw = None
        rospy.logdebug("Waiting for /camera/rgb/image_raw to be READY...")
        while self.camera_rgb_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_rgb_image_raw = rospy.wait_for_message("/camera/rgb/image_raw", Image, timeout=5.0)
                rospy.logdebug("Current /camera/rgb/image_raw READY=>")

            except:
                rospy.logerr("Current /camera/rgb/image_raw not ready yet, retrying for getting camera_rgb_image_raw")
        return self.camera_rgb_image_raw


    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.logdebug("Waiting for /kobuki/laser/scan to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message("/kobuki/laser/scan", LaserScan, timeout=5.0)
                rospy.logdebug("Current /kobuki/laser/scan READY=>")

            except:
                rospy.logerr("Current /kobuki/laser/scan not ready yet, retrying for getting laser_scan")
        return self.laser_scan


    def _pose_callback(self, data):
        self.odom = data

    def _vel_callback(self, data):
        self.velocity = data

    def _gt_pose_callback(self, data):
        self.gt_odom = data

    def _gt_neighbor_pose_callback(self, data):
        self.gt_neighbor_odom = data

    def _firefly_pose_callback(self, data):
        self.firefly_odom = data

    def _target_callback(self, data):
        self.target = data

    def _gt_target_callback(self, data):
        self.gt_target = data

    def _target_vel_callback(self, data):
        self.target_velocity = data

    def _camera_depth_image_raw_callback(self, data):
        self.camera_depth_image_raw = data

    def _camera_depth_points_callback(self, data):
        self.camera_depth_points = data

    def _camera_rgb_image_raw_callback(self, data):
        self.camera_rgb_image_raw = data

    def _laser_scan_callback(self, data):
        self.laser_scan = data

    def _detections_feedback_callback(self, data):
        self.feedback = data

    def _noisy_joints_callback(self,data):
        self.noisy_joints = data

    def _noisy_joints_neighbor_callback(self,data):
        self.noisy_joints_neighbor = data

    def _noisy_bbox_callback(self,data):
        self.noisy_bbox = data

    def _noisy_bbox_neighbor_callback(self,data):
        self.noisy_bbox_neighbor = data

    def _noisy_detection_callback(self,data):
        self.noisy_detection = data

    def _noisy_neighbor_detection_callback(self,data):
        self.noisy_neighbor_detection = data

    def _alphapose_callback(self, data):
        self.alphapose = data

    def _alphapose_neighbor_callback(self, data):
        self.alphapose_neighbor = data

    def _alphapose_bbox_callback(self, data):
        self.bbox = data

    def _alphapose_bbox_neighbor_callback(self, data):
        self.bbox_neighbor = data

    def _alphapose_detection_callback(self, data):
        self.detection = data

    def _alphapose_neighbor_detection_callback(self, data):
        self.detection_neighbor = data

    def _command_callback(self, data):
        self.command = data

    def _camera_info_callback(self, data):
        self.caminfo = data
        self.P1 = np.array([self.caminfo.P[0:4],self.caminfo.P[4:8],self.caminfo.P[8:12]])

    def _camera_info_neighbor_callback(self, data):
        self.caminfo_neighbor = data
        self.P2 = np.array([self.caminfo_neighbor.P[0:4],self.caminfo_neighbor.P[4:8],self.caminfo_neighbor.P[8:12]])

    def  _joints_gt_callback(self,data):
        self.joints_gt_data = data

    def  _mpc_command_callback(self,data):
        self.mpc_command_data = data


    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.create_circle(radius=5)
        machine_name = '/machine_'+str(self.robotID);
        self.command_topic = machine_name+"/command"
        self._cmd_vel_pub = rospy.Publisher(self.command_topic, uav_pose, queue_size=1)
        outPose = uav_pose()
        outPose.header.stamp = rospy.Time.now()
        outPose.header.frame_id="world"

        outPose.POI.x = 0
        outPose.POI.y = 0
        outPose.POI.z = 0

        r = 8
        t = np.random.choice(5,1);
        outPose.position.x = r*np.cos(self.theta[t[0]])
        outPose.position.y = r*np.sin(self.theta[t[0]])
        outPose.position.z = -r
        self._cmd_vel_pub.publish(outPose)

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_base(self, action, horizon=1, update_rate=100, init= False):
        """
        It will move the base based on the linear and angular velocties given.
        :param action: u_x, u_y, u_z
        :param horizon: Prediction horizon for MAV always(1)
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        import tf

        #use LTI dynamics with A = I with velocity control input
        # for k in range(horizon):
        #     outPose.position.x += outPose.velocity.x/update_rate
        #     outPose.position.y += outPose.velocity.y/update_rate
        
        outPose = uav_pose()
        stamp = rospy.Time.now()
        outPose.header.stamp = stamp
        outPose.header.frame_id="world"
        if init:
            outPose.POI.x = 0
            outPose.POI.y = 0
            outPose.POI.z = 0

            r = 8
            t = np.random.choice(5,1);

            outPose.position.x = r*np.cos(self.theta[t[0]])
            outPose.position.y = r*np.sin(self.theta[t[0]])
            outPose.position.z = -r

        else:
            try:
                (trans,rot) = self.listener.lookupTransform( 'world_ENU',self.rotors_machine_name+'/base_link', stamp)
            except:
                (trans,rot) = self.listener.lookupTransform( 'world_ENU',self.rotors_machine_name+'/base_link', rospy.Time(0))
                
            (r,p,y) = tf.transformations.euler_from_quaternion(rot)
            homogeneous_matrix = tf.transformations.euler_matrix(0,0,y,axes='sxyz')
            homogeneous_matrix[0:3,3] = trans


            gt_target = self.get_gt_target(stamp)
            outPose.POI.x = gt_target.pose.pose.position.x
            outPose.POI.y = gt_target.pose.pose.position.y
            outPose.POI.z = gt_target.pose.pose.position.z
  
            rospy.logwarn("END ACTION ==>"+str(action))

            gt_odom = self.get_gt_odom()

            act  = np.append(action[0:2],0)

            # Using MPC
            action_MPC = np.array([self.mpc_command_data.position.x-gt_odom.position.x,self.mpc_command_data.position.y-gt_odom.position.y, self.mpc_command_data.position.z-gt_odom.position.z])
            goal_MPC = np.array([self.mpc_command_data.position.x,self.mpc_command_data.position.y, self.mpc_command_data.position.z, 1])
            action_MPC = np.dot(np.linalg.inv(homogeneous_matrix),goal_MPC)
            if np.isnan(action_MPC).any():
                firefly_odom = self.get_firefly_odom()
                action_MPC=np.array([gt_odom.position.x,gt_odom.position.y,gt_odom.position.z])

            # goal = homogeneous_matrix.dot(np.concatenate((np.array([act[0]+action_MPC[0],act[1]+action_MPC[1],0]),np.array([1]))))                        
            goal = homogeneous_matrix.dot(np.concatenate((np.array([action_MPC[0],action_MPC[1],0]),np.array([1]))))                        
            rospy.logwarn("END ACTION==>"+str(action))

            #Publish Control Command with Fixed Height of 8m
            outPose.position.x = self.mpc_command_data.position.x#goal[0]
            outPose.position.y = self.mpc_command_data.position.y#goal[1]
            outPose.position.z = -5 

            rospy.logwarn("Current MAV Pose ==>x:"+str(gt_odom.position.x)+", y:"+str(gt_odom.position.y)+", z:"+str(gt_odom.position.z))


        rospy.logdebug("Firefly Command>>" + str(outPose))
        self._check_publishers_connection()
        
        self._cmd_vel_pub.publish(outPose)
        rate = rospy.Rate(update_rate)  
        try:
            rate.sleep()
        except:
            pass

    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate, min_laser_distance=-1):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        raise NotImplementedError()


    def has_crashed(self, min_laser_distance):
        """
        It states based on the laser scan if the robot has crashed or not.
        Crashed means that the minimum laser reading is lower than the
        min_laser_distance value given.
        If min_laser_distance == -1, it returns always false, because its the way
        to deactivate this check.
        """
        raise NotImplementedError()
        # robot_has_crashed = False
        #
        # if min_laser_distance != -1:
        #     laser_data = self.get_laser_scan()
        #     for i, item in enumerate(laser_data.ranges):
        #         if item == float ('Inf') or numpy.isinf(item):
        #             pass
        #         elif numpy.isnan(item):
        #            pass
        #         else:
        #             # Has a Non Infinite or Nan Value
        #             if (item < min_laser_distance):
        #                 rospy.logerr("TurtleBot HAS CRASHED >>> item=" + str(item)+"< "+str(min_laser_distance))
        #                 robot_has_crashed = True
        #                 break
        # return robot_has_crashed


    def get_odom(self):
        return self.odom

    def get_velocity(self,stamp=None):
        if stamp == None:
            stamp = rospy.Time.now()
        return self.velocity_cache.getElemBeforeTime(stamp)

    def get_velocity_neighbor(self,stamp=None):
        if stamp == None:
            stamp = rospy.Time.now()
        return self.velocity_neighbor_cache.getElemBeforeTime(stamp)

    def get_gt_odom(self):
        return self.gt_odom

    def get_neighbor_gt_odom(self,stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()
        # return self.gt_neighbor_cache.getElemBeforeTime(stamp)
        return self.gt_neighbor_odom

    def get_firefly_odom(self):
        return self.firefly_odom

    def get_target(self):
        return self.target

    def get_gt_target(self,stamp=None):
        if stamp == None:
            stamp = rospy.Time.now()
        return self.gt_target_cache.getElemBeforeTime(stamp)

    def get_target_velocity(self):
        return self.target_velocity

    def get_detections_feedback(self):
        return self.feedback

    def get_noisy_joints(self,stamp=None):
        if stamp == None:
            stamp = rospy.Time.now()
        return self.noisy_joints_cache.getElemBeforeTime(stamp)


    def get_noisy_joints_neighbor(self,stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()
        return self.noisy_joints_neighbor_cache.getElemBeforeTime(stamp)

    def get_noisy_bbox(self,stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()
        return self.noisy_bbox_cache.getElemBeforeTime(stamp)

    def get_noisy_bbox_neighbor(self,stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()
        return self.noisy_bbox_neighbor_cache.getElemBeforeTime(stamp)

    def get_noisy_detection(self):
        return self.noisy_detection

    def get_noisy_neighbor_detection(self):
        return self.noisy_neighbor_detection

    def get_alphapose(self,stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()
        return self.alphapose_cache.getElemBeforeTime(stamp)

    def get_alphapose_neighbor(self,stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()
        return self.alphapose_neighbor_cache.getElemBeforeTime(stamp)

    def get_alphapose_bbox(self,stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()
        return self.alphapose_bbox_cache.getElemBeforeTime(stamp)

    def get_alphapose_bbox_neighbor(self,stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()
        return self.alphapose_bbox_neighbor_cache.getElemBeforeTime(stamp)

    def get_mhmr_joints(self,stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()
        return self.mhmr_cache.getElemBeforeTime(stamp)

    def get_alphapose_detection(self):
        return self.detection

    def get_alphapose_detection_neighbor(self):
        return self.detection_neighbor

    def get_camera_depth_image_raw(self):
        return self.camera_depth_image_raw

    def get_camera_depth_points(self):
        return self.camera_depth_points

    def get_camera_rgb_image_raw(self):
        return self.camera_rgb_image_raw

    def get_laser_scan(self):
        return self.laser_scan

    def get_latest_command(self):
        return self.command

    def get_cam_intrinsic(self):
        return self.P1

    def get_neighbor_cam_intrinsic(self):
        return self.P2

    def get_joints_gt(self):
        return self.joints_gt_data

    def reinit_sensors(self):
        """
        This method is for the tasks so that when reseting the episode
        the sensors values are forced to be updated with the real data and

        """
