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

class FireflyEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for robot environment.
    """

    def __init__(self, ros_ws_abspath,**kwargs):
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
        # Variables that we give through the constructor.
        # None in this case

        # We launch the ROSlaunch that spawns the robot into the world
        # ROSLauncher(rospackage_name="turtlebot_gazebo",
        #             launch_file_name="put_robot_in_world.launch",
        #             ros_ws_abspath=ros_ws_abspath)


        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(FireflyEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD",**kwargs)

        self.pose_topic = "/machine_1/pose"

        self.velocity_topic = "/firefly_1/ground_truth/odometry"

        self.gt_pose_topic = "/machine_1/pose/groundtruth"

        self.gt_neighbor_pose_topic = "/machine_2/pose/groundtruth"

        self.firefly_pose_topic = "/firefly_1/ground_truth/pose_with_covariance"

        self.target_topic = "/machine_1/target_tracker/pose"

        self.gt_target_topic = "/actorpose"

        self.gt_target_vel_topic = "/actorvel"

        self.target_velocity_topic = "/machine_1/target_tracker/twist"

        self.command_topic = "/machine_1/command"

        self.neighbor_command_topic = "/machine_2/command"

        self.destination_topic = "/machine_1/destination"

        self.detections_feedback_topic = "/machine_1/object_detections/feedback"

        self.alphapose_topic = "/machine_1/result_alpha"

        self.alphapose_neighbor_topic = "/machine_2/result_alpha"

        self.alphapose_bbox_topic = "/machine_1/result_bbox"

        self.alphapose_bbox_neighbor_topic = "/machine_2/result_bbox"

        self.alphapose_detection = "/machine_1/detection"

        self.alphapose_neighbor_detection = "/machine_2/detection"

        self.camera_info = "/firefly_1/firefly_1/xtion/rgb/camera_info"

        self.camera_info_neighbor = "/firefly_2/firefly_2/xtion/rgb/camera_info"

        self.joints_gt = '/gt_joints'



        import tf
        self.listener = tf.TransformListener()




        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()



        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber(self.firefly_pose_topic,PoseWithCovarianceStamped , self._firefly_pose_callback)
        rospy.Subscriber(self.pose_topic, uav_pose, self._pose_callback)
        rospy.Subscriber(self.velocity_topic, Odometry, self._vel_callback)
        rospy.Subscriber(self.gt_pose_topic, uav_pose, self._gt_pose_callback)
        rospy.Subscriber(self.gt_neighbor_pose_topic, uav_pose, self._gt_neighbor_pose_callback)
        rospy.Subscriber(self.target_topic, PoseWithCovarianceStamped, self._target_callback)
        rospy.Subscriber(self.gt_target_topic, Odometry, self._gt_target_callback)
        rospy.Subscriber(self.target_velocity_topic, TwistWithCovarianceStamped, self._target_vel_callback)
        rospy.Subscriber(self.detections_feedback_topic, NeuralNetworkFeedback, self._detections_feedback_callback)
        rospy.Subscriber(self.alphapose_topic, AlphaRes, self._alphapose_callback)
        rospy.Subscriber(self.alphapose_neighbor_topic, AlphaRes, self._alphapose_neighbor_callback)
        rospy.Subscriber(self.alphapose_bbox_topic, AlphaRes, self._alphapose_bbox_callback)
        rospy.Subscriber(self.alphapose_bbox_neighbor_topic, AlphaRes, self._alphapose_bbox_neighbor_callback)
        rospy.Subscriber(self.alphapose_detection, Int16, self._alphapose_detection_callback)
        rospy.Subscriber(self.alphapose_neighbor_detection, Int16, self._alphapose_neighbor_detection_callback)
        rospy.Subscriber(self.command_topic, uav_pose, self._command_callback)
        rospy.Subscriber(self.camera_info, CameraInfo, self._camera_info_callback)
        rospy.Subscriber(self.camera_info_neighbor, CameraInfo, self._camera_info_neighbor_callback)
        rospy.Subscriber(self.joints_gt, PoseArray, self._joints_gt_callback)

        self._dest_vel_pub = rospy.Publisher(self.destination_topic, PoseStamped, queue_size=1)
        self._cmd_vel_pub = rospy.Publisher(self.command_topic, uav_pose, queue_size=1)
        self._cmd_neighbor_vel_pub = rospy.Publisher(self.neighbor_command_topic, uav_pose, queue_size=1)
        self._target_init_pub = rospy.Publisher(self.target_topic, PoseWithCovarianceStamped, queue_size=1)

        self._check_all_sensors_ready()
        self._check_publishers_connection()

        self.create_circle(radius=5)

        # self.gazebo.pauseSim()

        rospy.logdebug("Finished FireflyEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------


    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL MODULES READY")
        self._check_odom_ready()
        # self._check_target_ready()
        self._check_gt_target_ready()
        # self._check_target_vel_ready()
        # self._check_feedback_ready()
        self._check_alphapose_ready()
        # self._check_neighbor_alphapose_ready()
        # self._check_neighbor_ready()
        self._check_camera_rgb_image_info_ready()
        # self._check_camera_neighbor_rgb_image_info_ready()

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

    def _check_alphapose_ready(self):
        self.alphapose = None
        rospy.logdebug("Waiting for /alphapose to be READY...")
        while self.alphapose is None and not rospy.is_shutdown():
            try:
                self.alphapose = rospy.wait_for_message(self.alphapose_topic, AlphaRes, timeout=5.0)
                rospy.logdebug("Current /alphapose READY=>")

            except:
                rospy.logerr("Current /alphapose not ready yet, retrying for getting target")

        return self.alphapose

    def _check_neighbor_alphapose_ready(self):
        self.alphapose_neighbor = None
        rospy.logdebug("Waiting for neighbor /alphapose to be READY...")
        while self.alphapose_neighbor is None and not rospy.is_shutdown():
            try:
                self.alphapose_neigbor = rospy.wait_for_message(self.alphapose_neighbor_topic, AlphaRes, timeout=5.0)
                rospy.logdebug("Current /alphapose READY=>")

            except:
                rospy.logerr("Current neighbor /alphapose not ready yet, retrying for getting target")

        return self.alphapose_neighbor

    def _check_neighbor_ready(self):
        self.gt_neighbor_odom = None
        rospy.logdebug("Waiting for /machine_2/pose to be READY...")
        while self.gt_neighbor_odom is None and not rospy.is_shutdown():
            try:
                self.gt_neighbor_odom = rospy.wait_for_message(self.gt_neighbor_pose_topic, uav_msgs, timeout=5.0)
                rospy.logdebug("Current /machine_2/pose READY=>")

            except:
                rospy.logerr("Current  /machine_2/pose not ready yet, retrying for getting target")

        return self.alphapose

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
        raise NotImplementedError()

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

    def create_circle(self, radius=5):
        self.theta = [k for k in np.arange(0,2*np.pi,0.1)]
        x = radius*np.cos(self.theta)
        y = radius*np.sin(self.theta)
        self.init_circle = [x,y]

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_base(self, action, horizon, update_rate=100, init= False):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param action: u_x, u_y, u_z
        :param horizon: Prediction horizon for MAV
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        import tf


        # outPose.velocity.x = self.odom.velocity.x
        # outPose.velocity.y = self.odom.velocity.y
        # outPose.velocity.z = self.odom.velocity.z

        #use LTI dynamics with A = I with acceleration control input
        # for k in range(horizon):
        #     outPose.velocity.x += action[0]/update_rate**2
        #     outPose.position.x += outPose.velocity.x/update_rate
        #     outPose.velocity.y += action[1]/update_rate**2
        #     outPose.position.y += outPose.velocity.y/update_rate
        #     outPose.velocity.z += (-1)*action[2]/update_rate**2
        #     outPose.position.z += outPose.velocity.z/update_rate
        #use LTI dynamics with A = I with velocity control input
        outPose = uav_pose()
        outPose.header.stamp = rospy.Time.now()
        outPose.header.frame_id="world"
        if init:
            outPose.POI.x = 0
            outPose.POI.y = 0
            outPose.POI.z = 0
            x = np.random.choice(63,1);y = np.random.choice(63,1)
            r = np.random.randint(4,8);t = np.random.choice(63,1);
            outPose.position.x = r*np.cos(self.theta[t[0]])
            outPose.position.y = r*np.sin(self.theta[t[0]])
            outPose.position.z = -r
            self._cmd_neighbor_vel_pub.publish(outPose)
            # x = np.random.choice(63,1);y = np.random.choice(63,1)
            # r = np.random.randint(4,8);t = np.random.choice(63,1);
            outPose.position.x = r*np.cos(self.theta[t[0]])
            outPose.position.y = r*np.sin(self.theta[t[0]])
            outPose.position.z = -r

            # outPose.position.x = -6#r*np.cos(self.theta[t[0]])
            # outPose.position.y = 0#r*np.sin(self.theta[t[0]])
            # outPose.position.z = -6#-r

        else:

            #initialize outpose with current position and velocity

            '''estimated target position'''
            # outPose.POI.x = self.target.pose.pose.position.x
            # outPose.POI.y = self.target.pose.pose.position.y
            # outPose.POI.z = self.target.pose.pose.position.z

            '''ground truth target position'''
            outPose.POI.x = self.gt_target.pose.pose.position.x
            outPose.POI.y = self.gt_target.pose.pose.position.y
            outPose.POI.z = self.gt_target.pose.pose.position.z

            # pointstamp = PointStamped()
            # pointstamp.header.frame_id = 'firefly_1/base_link'
            # pointstamp.header.stamp = rospy.Time(0)
            # pointstamp.point.x = action[0]
            # pointstamp.point.y = action[1]
            # pointstamp.point.z = action[2]
            # rospy.logwarn("END ACTION PRETRAIN==>"+str(self.act_pretrained))
            # rospy.logwarn("END ACTION PRETRAIN WITH JOINTS==>"+str(self.act_pretrained_joints))
            rospy.logwarn("END ACTION ==>"+str(action))
            # outPose.velocity.x = action[0]
            # outPose.velocity.y = action[1]
            # outPose.velocity.z = action[2]

            (trans,rot) = self.listener.lookupTransform( 'world_ENU','firefly_1/base_link', rospy.Time(0))
            (r,p,y) = tf.transformations.euler_from_quaternion(rot)
            homogeneous_matrix = tf.transformations.euler_matrix(0,0,y,axes='sxyz')
            homogeneous_matrix[0:3,3] = trans
            #  print(homogeneous_matrix) #for debug
            if self._MAV_out_ws:
                outPose.position.x = 0
                outPose.position.y = 0
                outPose.position.z = self.desired_distance


            else:
                goal = homogeneous_matrix.dot(np.concatenate((np.array(action[0:3]),np.array([1]))))
                # act = self.listener.transformPoint('world_ENU',pointstamp)
                # outPose.pose.position.x = goal[0] #act.point.x
                # outPose.pose.position.y = goal[1]
                # outPose.pose.position.z = goal[2]

                '''controlling robot yaw'''
                # yaw = y + action[3]
                # outPose.POI.x = trans[0]+np.cos(yaw)
                # outPose.POI.y = trans[1]+np.sin(yaw)
                # outPose.POI.z = 1


                outPose.position.x = goal[0]
                outPose.position.y = goal[1]
                outPose.position.z = goal[2]

            firefly_odom = self.get_firefly_odom()
            gt_odom = self.get_gt_odom()
            # rospy.logwarn("END Transformed Goal ==>x:"+str(act.point.x)+", y:"+str(act.point.y)+", z:"+str(act.point.z))
            # rospy.logwarn("END Transformed Goal ==>x:"+str(outPose.position.x)+", y:"+str(outPose.position.y)+", z:"+str(outPose.position.z))
            rospy.logwarn("Current MAV Pose ==>x:"+str(gt_odom.position.x)+", y:"+str(gt_odom.position.y)+", z:"+str(gt_odom.position.z))
            # rospy.logwarn("Current MAV Heading ==>"+ str(yaw))
            # rospy.logwarn("Current MAV Control ==>x:"+str(outPose.position.x - gt_odom.position.x)+", y:"+str(outPose.position.y - gt_odom.position.y)+", z:"+str(outPose.position.z - gt_odom.position.z))
            # rospy.logwarn("Current Firefly Pose ==>x:"+str(firefly_odom.pose.pose.position.x)+", y:"+str(firefly_odom.pose.pose.position.y)+", z:"+str(firefly_odom.pose.pose.position.z))


            # pointstamp = PointStamped()
            # pointstamp.header.frame_id = 'world_ENU'
            # pointstamp.header.stamp = rospy.Time(0)
            # pointstamp.point.x = act.point.x
            # pointstamp.point.y = act.point.y
            # pointstamp.point.z = -act.point.z
            # act = self.listener.transformPoint('firefly_1/base_link',pointstamp)
            # rospy.logwarn("Goal Local ==>x:"+str(act.point.x)+", y:"+str(act.point.y)+", z:"+str(-act.point.z))

        rospy.logdebug("Firefly Command>>" + str(outPose))
        self._check_publishers_connection()
        #publish desired position and velocity
        self._cmd_vel_pub.publish(outPose)
        rate = rospy.Rate(update_rate)  # 10hz
        rate.sleep()
        #time.sleep(0.02)
        """
        self.wait_until_twist_achieved(cmd_vel_value,
                                        epsilon,
                                        update_rate,
                                        min_laser_distance)
        """


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

    def get_velocity(self):
        return self.velocity

    def get_gt_odom(self):
        return self.gt_odom

    def get_neighbor_gt_odom(self):
        return self.gt_neighbor_odom

    def get_firefly_odom(self):
        return self.firefly_odom

    def get_target(self):
        return self.target

    def get_gt_target(self):
        return self.gt_target

    def get_target_velocity(self):
        return self.target_velocity

    def get_detections_feedback(self):
        return self.feedback

    def get_alphapose(self):
        return self.alphapose

    def get_alphapose_neighbor(self):
        return self.alphapose_neighbor

    def get_alphapose_bbox(self):
        return self.bbox

    def get_alphapose_bbox_neighbor(self):
        return self.bbox_neighbor

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
