import numpy as np
import rospy
import moveit_commander
import cv2
import tf
import sys
from cv_bridge import CvBridge, CvBridgeError
from openai_ros import robot_gazebo_env
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Quaternion, Vector3
from openai_ros.openai_ros_common import ROSLauncher


class MyCobotEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, ros_ws_abspath):
        """
        Initializes a new MyCobotEnv environment.
        MyCobot doesnt use controller_manager, therefore we wont reset the
        controllers in the standard fashion. For the moment we wont reset them.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered usefull for AI learning.

        Sensor Topic List:
        * /depth_camera/color/image_raw: RGB image of the depth sensor
        * /depth_camera/depth/image_raw: 2d Depth image of the depth sensor

        Actuators Topic List:

        Args:
        """
        rospy.logdebug("Start MyCobotEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # We launch the ROSlaunch that spawns the robot into the world
        # ROSLauncher(rospackage_name="mycobot_moveit",
        #             launch_file_name="mycobot_moveit_gazebo.launch",
        #             ros_ws_abspath=ros_ws_abspath)

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(MyCobotEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="SIMULATION")

        self.gazebo.unpauseSim()
        
        self.bridge = CvBridge() # For RGB, Depth image

        self._check_all_sensors_ready()

        moveit_commander.roscpp_initialize(sys.argv)
        self.move_group = moveit_commander.MoveGroupCommander("mycobot_arm", wait_for_servers=0)

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/depth_camera/color/image_raw", Image, self._rgb_img_callback, queue_size=1)
        rospy.Subscriber("/depth_camera/depth/image_raw", Image, self._depth_img_callback, queue_size=1)
        # self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # self._check_publishers_connection()

        self.gazebo.pauseSim()

        rospy.logdebug("Finished MyCobotEnv INIT...")

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
        rospy.logdebug("START ALL SENSORS READY")
        self._check_rgb_img_ready()
        self._check_depth_img_ready()
        rospy.logdebug("ALL SENSORS READY")
    
    def _check_rgb_img_ready(self):
        self.rgb_image = None
        rospy.logdebug("Waiting for /depth_camera/color/image_raw to be READY...")
        while self.rgb_image is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/depth_camera/color/image_raw", Image, timeout=5.0)
                self._rgb_img_callback(data)
                rospy.logdebug("Current /depth_camera/color/image_raw READY=>")
            except:
                rospy.logerr("Current /depth_camera/color/image_raw not ready yet, retrying for getting rgb_img")

        return self.rgb_image
    
    def _check_depth_img_ready(self):
        self.depth_image = None
        rospy.logdebug("Waiting for /depth_camera/depth/image_raw to be READY...")
        while self.depth_image is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/depth_camera/depth/image_raw", Image, timeout=5.0)
                self._depth_img_callback(data)
                rospy.logdebug("Current /depth_camera/depth/image_raw READY=>")
            except:
                rospy.logerr("Current /depth_camera/depth/image_raw not ready yet, retrying for getting depth_img")

        return self.depth_image

    def _rgb_img_callback(self, data):
        try:
            input_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        except CvBridgeError as e:
            rospy.logerr(e)
    
    def _depth_img_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            rospy.logerr(e)
        
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

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_angle_goal(self, angle_goal_deg):
        """目標関節角度へ動かす

        Args:
            angle_goal_deg (ndarray): 目標関節角度、degree
        """
        rospy.logdebug("MyCobot Target Joint>> ")
        rospy.logdebug(angle_goal_deg.astype("str"))
        # 関節の角度でゴール状態を指定
        self.move_group.set_joint_value_target(np.deg2rad(angle_goal_deg))

        # モーションプランの計画と実行
        self.move_group.go(wait=True)

        # 後処理
        self.move_group.stop()
        self.calc_pose_and_angles()
    
    def move_pose_goal(self, pose_goal):
        """目標姿勢へ動かす

        Args:
            pose_goal (ndarray): 目標姿勢、前半3つが位置、後半3つがオイラー角（rad）
        """
        rospy.logdebug("MyCobot Target Pose>>")
        rospy.logdebug(pose_goal.astype("str"))
        # エンドエフェクタの姿勢でゴール状態を指定
        movegroup_pose_goal = Pose()
        movegroup_pose_goal.position = Vector3(pose_goal[0], pose_goal[1], pose_goal[2])
        q = tf.transformations.quaternion_from_euler(pose_goal[3], pose_goal[4], pose_goal[5])
        movegroup_pose_goal.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.move_group.set_pose_target(movegroup_pose_goal)

        # モーションプランの計画と実行
        self.move_group.go(wait=True)

        # 後処理
        self.move_group.stop()
        self.calc_pose_and_angles()

    def get_rgb_img(self):
        return self.rgb_image

    def get_depth_img(self):
        return self.depth_image
    
    def calc_pose_and_angles(self):
        if not self.gazebo.is_pause:
            ee_pose = self.move_group.get_current_pose().pose
            ee_pos = np.array([ee_pose.position.x, ee_pose.position.y, ee_pose.position.z])
            ee_eul = tf.transformations.euler_from_quaternion([ee_pose.orientation.x,
                                                            ee_pose.orientation.y,
                                                            ee_pose.orientation.z,
                                                            ee_pose.orientation.w],
                                                            axes='sxyz')
            self.pose = np.concatenate([ee_pos, ee_eul])
            self.angles = self.move_group.get_current_joint_values()
    
    def get_pose(self):
        return self.pose
    
    def get_angles(self):
        return self.angles
