##! /usr/bin/env python
import numpy
import rospy
from tf2.transformations import quaternion_from_euler
from gazebo_msgs.srv import GetModelState, GetWorldProperties
from openai_ros import robot_gazebo_env_goal
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState, Image, PointCloud2
from nav_msgs.msg import Odometry
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest, GetPositionIK, GetPositionIKRequest
from abb_catkin.srv import EePose, EePoseRequest, EeRpy, EeRpyRequest, EeTraj, EeTrajRequest, JointTraj, JointTrajRequest
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
import actionlib

class Abbenv(robot_gazebo_env_goal.RobotGazeboEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(self):
        print ("Entered ABB Env")
        """Initializes a new Fetch environment.

        Args:
            
        """


        """
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that the stream of data doesn't flow. This is for simulations
        that are paused for whatever reason
        2) If the simulation was running already for some reason, we need to reset the controllers.
        This has to do with the fact that some plugins with tf don't understand the reset of the simulation
        and need to be reset to work properly.
        """

        # We Start all the ROS related Subscribers and publishers
        
        JOINT_STATES_SUBSCRIBER = '/joint_states'
        
        self.joint_states_sub = rospy.Subscriber(JOINT_STATES_SUBSCRIBER, JointState, self.joints_callback)
        self.joints = JointState()
        
        #to ensure topics of camera are initialised

        image_raw_topic = '/rgb/image_raw'

        self.image_raw_sub = rospy.Subscriber(image_raw_topic, Image, self.image_raw_callback)
        self.image_raw = Image()

        depth_raw_topic = '/depth/image_raw'
      
        self.depth_raw_sub = rospy.Subscriber(depth_raw_topic, Image, self.depth_raw_callback)
        self.depth_raw = Image()

        depth_points_topic = '/depth/points'
      
        self.depth_points_sub = rospy.Subscriber(depth_points_topic, PointCloud2, self.depth_points_callback)
        self.depth_points = PointCloud2()

        #intializing important clients

        self.model_state_client = rospy.ServiceProxy('/get_model_state',GetModelState)
        self.world_properties_client = rospy.ServiceProxy('/get_world_properties', GetWorldProperties)
        self.ee_traj_client = rospy.ServiceProxy('/ee_traj_srv', EeTraj)
        self.joint_traj_client = rospy.ServiceProxy('/joint_traj_srv', JointTraj)
        self.ee_pose_client = rospy.ServiceProxy('/ee_pose_srv', EePose)
        self.ee_rpy_client = rospy.ServiceProxy('/ee_rpy_srv', EeRpy)
        self.joint_state_valid_client = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        self.joint_state_from_pose_client = rospy.ServiceProxy('/GetPositionIK', GetPositionIK)

        #initializing action server for gripper passant add action clinet
        self.gripper_client = actionlib.SimpleActionClient('/gripper_controller/gripper_cmd', GripperCommandAction)



        # Variables that we give through the constructor.

        #self.controllers_list = []
        self.controllers_list = ["joint_state_controller", "arm_controller"]
        #self.controllers_list = ["joint_state_controller"]

        self.robot_name_space = ""
        
        # We launch the init function of the Parent Class robot_gazebo_env_goal.RobotGazeboEnv
        super(Abbenv, self).__init__(controllers_list=self.controllers_list,
                                          robot_name_space=self.robot_name_space,
                                          reset_controls=False) #False
        print("Exit ABB Env")



    # RobotGazeboEnv virtual methods
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    # FetchEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        self._check_joint_states_ready()
        #self._check_image_raw_ready() 
        #self._check_depth_raw_ready()
        #self._check_depth_points_ready()
        #self.check_gripper_ready()
        rospy.logdebug("ALL SENSORS READY")

    def check_gripper_ready(self):
        rospy.logdebug("Waiting for gripper action server to be ready")
        self.gripper_client.wait_for_server()
        rospy.logdebug("gripper action server is READY")

    def _check_joint_states_ready(self):
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message("/joint_states", JointState, timeout=5.0)
                rospy.logdebug("Current /joint_states READY=>" + str(self.joints))

            except:
                rospy.logerr("Current /joint_states not ready yet, retrying for getting joint_states")
        return self.joints


    def _check_image_raw_ready(self):
        self.image_raw = None
        while self.image_raw is None and not rospy.is_shutdown():
            try:
                self.image_raw = rospy.wait_for_message('/rgb/image_raw', Image, timeout=1.0)
                rospy.logdebug("Current /rgb/image_raw=>" + str(self.image_raw))

            except:
                rospy.logerr("Current /rgb/image_raw not ready yet, retrying for getting rgb/image_raw")
        return self.image_raw

    
    def _check_depth_raw_ready(self):
        self.depth_raw = None
        while self.depth_raw is None and not rospy.is_shutdown():
            try:
                self.depth_raw = rospy.wait_for_message('/depth/image_raw', Image, timeout=1.0)
                rospy.logdebug("Current /depth/image_raw=>" + str(self.depth_raw))

            except:
                rospy.logerr("Current /depth/image_raw not ready yet, retrying for getting depth/image_raw")
        return self.depth_raw


    def _check_depth_points_ready(self):
        self.depth_points = None
        while self.depth_points is None and not rospy.is_shutdown():
            try:
                self.depth_points = rospy.wait_for_message('/depth/points', PointCloud2, timeout=1.0)
                rospy.logdebug("Current /depth/points=>" + str(self.depth_points))

            except:
                rospy.logerr("Current /depth/points not ready yet, retrying for getting depth/points")
        return self.depth_points


    def joints_callback(self, data):
        self.joints = data

    def image_raw_callback(self, data):
        self.image_raw = data

    def depth_raw_callback(self, data):
        self.depth_raw = data

    def depth_points_callback(self, data):
        self.depth_points = data

    def get_joints(self):
        return self.joints

    def set_trajectory_ee(self, action):
        """
        Helper function.
        Wraps an action vector of joint angles into a JointTrajectory message.
        The velocities, accelerations, and effort do not control the arm motion
        """
        # Set up a trajectory message to publish. for the end effector
        
        ee_target = EeTrajRequest()
        ee_target.pose.orientation.w = quaternion_from_euler([0, 1.571, action[3]])
        ee_target.pose.position.x = action[0]
        ee_target.pose.position.y = action[1]
        ee_target.pose.position.z = action[2]
        result = self.ee_traj_client(ee_target)

        goal = GripperCommandGoal()
        goal.command.position = 0.8 if action[4:5] == [1, 0] else goal.command.position = 0
        goal.command.max_effort = -1.0  #THIS NEEDS TO BE CHANGEDDDDDD
        self.gripper_client.send_goal(goal)

        self.gripper_client.wait_for_result()

        return True

    def set_trajectory_joints(self, initial_qpos):
        """
        Helper function.
        Wraps an action vector of joint angles into a JointTrajectory message.
        The velocities, accelerations, and effort do not control the arm motion
        """
        # Set up a trajectory message to publish.
        
        joint_point = JointTrajRequest()
        
        joint_point.point.positions = [None] * 6
        joint_point.point.positions[0] = initial_qpos["joint0"]
        joint_point.point.positions[1] = initial_qpos["joint1"]
        joint_point.point.positions[2] = initial_qpos["joint2"]
        joint_point.point.positions[3] = initial_qpos["joint3"]
        joint_point.point.positions[4] = initial_qpos["joint4"]
        joint_point.point.positions[5] = initial_qpos["joint5"]
        #joint_point.point.positions[6] = initial_qpos["joint6"]
        
        result = self.joint_traj_client(joint_point)
        
        return result
   
    def get_ee_pose(self):

        #get the ee pose
        gripper_pose_req = EePoseRequest()
        gripper_pose = self.ee_pose_client(gripper_pose_req)

        #get gripper state in addition to state of success in command
        result = self.gripper_client.get_result()
        gripper_open = 0 if result.position > 0.0 else gripper_open = 1
        gripper_state = [gripper_open, result.reached_goal]

        return gripper_pose, gripper_state
        
    def get_ee_rpy(self):

        gripper_rpy_req = EeRpyRequest()
        gripper_rpy = self.ee_rpy_client(gripper_rpy_req)
        
        return gripper_rpy

    def get_available_models(self):
        world_properties = self.world_properties_client()
        return world_properties.model_names

    def get_model_states(self):

        #getting available model names
        #changing the data got from the client can help in getting the velocities of the objects also
        model_names = self.get_available_models()

        self.model_states = {model: self.model_state_client(model).pose.position for model in model_names}


    def check_ee_valid_pose(self,action):
        #checking the validity of the end effector pose
        #converting to joint state using ik and then getting the validity
        GPIK_Request =GetPositionIKRequest()
        GPIK_Request.group_name = GPIK_Request.group_name = 'arm_controller'
        GPIK_Request.robot_state.joint_state = self.joints
        GPIK_Request.avoid_collisions = True
        # this poses is related to the reference frame of gazebo
        # the pose is set as a radian value between 1.571 and -1.571
        GPIK_Request.pose_stamped.pose.position.x = action[ 0 ]
        GPIK_Request.pose_stamped.pose.position.y = action[ 1 ]
        GPIK_Request.pose_stamped.pose.position.z = action[ 2 ]
        GPIK_Request.pose_stamped.pose.orientation = quaternion_from_euler([0, 1.571, action[3]])
        GPIK_Response = self.joint_state_from_pose_client(GPIK_Request)
        if GPIK_Response.error_code == 1:
            return True
        else:
            return GPIK_Response.error_code

        # GSV_Request = GetStateValidityRequest()
        # GSV_Request.group_name = GPIK_Request.group_name ='arm_controller'
        # GSV_Request.robot_state.joint_state.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6' ]
        # GSV_Request.robot_state.joint_state.position = [ 0, 0, 0, 0, 0, 0 ]
        # valid = self.joint_state_valid_client(GSV_Request)
        # return valid


    # ParticularEnv methods
    # ----------------------------

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()
        #This should include intilization of different objects in the env getting their poses using
        #the get_model_states
        # publish topics l bt3mel randomize
        # a5od l position bta3 l objects b3d l randomize



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
