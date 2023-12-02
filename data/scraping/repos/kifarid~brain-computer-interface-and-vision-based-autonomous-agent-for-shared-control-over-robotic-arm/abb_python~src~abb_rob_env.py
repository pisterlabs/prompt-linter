#! /usr/bin/env python

import numpy as np
import rospy
import tf
from geometry_msgs.msg import PointStamped, PoseStamped
import sys
#from gym.envs.robotics.rotations import euler2quat
from gazebo_msgs.srv import GetModelState, GetWorldProperties
sys.path.insert(0, '/home/arl/env/src')
from openai_ros import robot_gazebo_env_goal
from std_msgs.msg import Float64
import sensor_msgs
from sensor_msgs.msg import JointState, Image, PointCloud2
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest, GetPositionIK, GetPositionIKRequest
from abb_catkin.srv import EePose, EePoseRequest, EeRpy, EeRpyRequest, EeTraj, EeTrajRequest, JointTraj, JointTrajRequest
from abb_catkin.srv import EePoseResponse, EeRpyResponse, EeTrajResponse, JointTrajResponse
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
import actionlib
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Empty
import ros_numpy
import moveit_commander
import geometry_msgs.msg
import trajectory_msgs.msg
import moveit_msgs.msg


class Abbenv(robot_gazebo_env_goal.RobotGazeboEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(self):
        #print ("Entered ABB Env")
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

        self.listener = tf.TransformListener()
        self.world_point = PointStamped()
        self.world_point.header.frame_id = "world"
        self.camera_point = PointStamped()

        # We Start all the ROS related Subscribers and publishers
        self.model_names = [ "obj0", "obj1", "obj2", "obj3", "obj4" ]
        JOINT_STATES_SUBSCRIBER = '/joint_states'
        
        self.joint_states_sub = rospy.Subscriber(JOINT_STATES_SUBSCRIBER, JointState, self.joints_callback)
        self.joints = JointState()
        
        #to ensure topics of camera are initialised

        image_raw_topic = '/rgb/image_raw'

        self.image_raw_sub = rospy.Subscriber(image_raw_topic, Image, self.image_raw_callback)
        self.image_raw = None

        depth_raw_topic = '/depth/image_raw'
      
        self.depth_raw_sub = rospy.Subscriber(depth_raw_topic, Image, self.depth_raw_callback)
        self.depth_raw = None

        #Camera parameters subscriber
        camera_param_sub = rospy.Subscriber('/rgb/camera_info', sensor_msgs.msg.CameraInfo, self.camera_param_callback)
        self.camera_param = None

        #initializing the domain randomization
        Delete_Model_Publisher = '/randomizer/delete'
        Spawn_Model_Publisher = '/randomizer/spawn'
        Randomize_Environment_Publisher = '/randomizers/randomizer/trigger'

        self.delete_objects = rospy.Publisher(Delete_Model_Publisher, Empty, queue_size=1)
        self.randomize_env = rospy.Publisher(Randomize_Environment_Publisher, Empty, queue_size=1)
        self.spawn_objects = rospy.Publisher(Spawn_Model_Publisher, Empty, queue_size=1)

        #intializing important clients and waiting for them to be alive
        rospy.wait_for_service('/gazebo/get_model_state')
        self.model_state_client = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        rospy.wait_for_service('/gazebo/get_world_properties')
        self.world_properties_client = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        rospy.wait_for_service('/check_state_validity')
        self.joint_state_valid_client = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        rospy.wait_for_service('/compute_ik')
        self.joint_state_from_pose_client = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        #rospy.wait_for_service('/arm_controller/query_state')


        #moveit python interface setup
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("manipulator")
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory)
        self.pose_target = geometry_msgs.msg.Pose()
        
        #additional part for avoiding camer collision

        rospy.sleep(2)

        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = 0.  #3deltaha hna
        p.pose.position.y = -0.47
        p.pose.position.z = 0.5
        self.scene.add_box("our_stereo", p, (0.2, 0.2, 0.1))


        #rospy.wait_for_service('/ee_traj_srv')
        #self.ee_traj_client = rospy.ServiceProxy('/ee_traj_srv', EeTraj)
        #rospy.wait_for_service('/joint_traj_srv')
        #self.joint_traj_client = rospy.ServiceProxy('/joint_traj_srv', JointTraj)
        #rospy.wait_for_service('/ee_pose_srv')
        #self.ee_pose_client = rospy.ServiceProxy('/ee_pose_srv', EePose)
        #rospy.wait_for_service('/ee_rpy_srv')
        #self.ee_rpy_client = rospy.ServiceProxy('/ee_rpy_srv', EeRpy)

        #initializing action server for gripper passant add action client
        self.gripper_client = actionlib.SimpleActionClient('/gripper_controller/gripper_cmd', GripperCommandAction)



        # Variables that we give through the constructor.
        self.controllers_list = ["joint_state_controller", "arm_controller","gripper_controller"]   #hna fy e5tlaf

        self.robot_name_space = ""
        
        # We launch the init function of the Parent Class robot_gazebo_env_goal.RobotGazeboEnv
        super(Abbenv, self).__init__(controllers_list=self.controllers_list,
                                          robot_name_space=self.robot_name_space,
                                          reset_controls=False) #False
        #print("Exit ABB Env")



    # RobotGazeboEnv virtual methods
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        #print('bad2na check')
        rospy.wait_for_service('/arm_controller/query_state')
        rospy.wait_for_service('/gazebo/get_model_state')
	rospy.wait_for_service('/gazebo/get_world_properties')
        rospy.wait_for_service('/check_state_validity')
        rospy.wait_for_service('/compute_ik')
        #print('fadel l sense fl check')
        self._check_all_sensors_ready()
        return True


    # FetchEnv virtual methods
    # ----------------------------
    #leh commented ba2y el checks?
    def _check_all_sensors_ready(self):
	print('check sensor start')
        self._check_joint_states_ready() ##hna fy e5tlaf
        self._check_image_raw_ready() 
        self.check_gripper_ready()
        self._check_depth_raw_ready()
        #self._check_depth_points_ready()   
        print('all sensors ready')        
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

    def _check_camera_param_ready(self):
        self.camera_param = None
        while self.camera_param is None and not rospy.is_shutdown():
            try:
                self.camera_param = rospy.wait_for_message('/rgb/camera_info', sensor_msgs.msg.CameraInfo, timeout=1.0)
                rospy.logdebug("Current /rgb/camera_info=>" + str(self.camera_param))

            except:
                rospy.logerr("Current /rgb/camera_info not ready yet, retrying for getting rgb/camera_info")
        return self.camera_param

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

    def camera_param_callback(self, data):
        self.camera_param = data

    def joints_callback(self, data):
        self.joints = data

    def image_raw_callback(self, data):
        data = ros_numpy.numpify(data)
        self.image_raw = data

    def depth_raw_callback(self, data):
        data = ros_numpy.numpify(data)
        data = np.nan_to_num(data) #remove nan values in depth image
        self.depth_raw = data

    def get_stacked_image(self):#addition of depth to image

        full_image = np.dstack((self.image_raw, self.depth_raw))
        return full_image

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
        quat = quaternion_from_euler(0, 1.571, action[3], 'sxyz')
        ee_target.pose.orientation.x = quat[0]
        ee_target.pose.orientation.y = quat[1]
        ee_target.pose.orientation.z = quat[2]
        ee_target.pose.orientation.w = quat[3]
        ee_target.pose.position.x = action[0]
        ee_target.pose.position.y = action[1]
        ee_target.pose.position.z = action[2]

        if self.check_ee_valid_pose(action):
            result = self.ee_traj_callback(ee_target)

        goal = GripperCommandGoal()
        goal.command.position = 0.8 if action[4] >= 0.0 else 0.0
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
        
        result = self.joint_traj_callback(joint_point)
        return result
   
    def get_ee_pose(self):

        #get the ee pose
        gripper_pose_req = EePoseRequest()
        gripper_pose = self.ee_pose_callback(gripper_pose_req)

        #get gripper state in addition to state of success in command
        result = self.gripper_client.get_result()
        gripper_open = 0 if result.position > 0.0 else 1
        gripper_state = [int(gripper_open), 1]

        return gripper_pose, gripper_state
        
    def get_ee_rpy(self):

        gripper_rpy_req = EeRpyRequest()
        gripper_rpy = self.ee_rpy_callback(gripper_rpy_req)
        
        return gripper_rpy

    def get_available_models(self):
        world_properties = self.world_properties_client()
        return world_properties.model_names

    def get_model_states(self):

        #getting available model names
        #changing the data got from the client can help in getting the velocities of the objects also
        model_states = { model:  np.around(ros_numpy.numpify((self.model_state_client(model,'')).pose.position), decimals = 3)  for model in self.model_names } #hna fy e5tlaf

        return model_states



    def check_ee_valid_pose(self,action):
        #checking the validity of the end effector pose
        #converting to joint state using ik and then getting the validity
        GPIK_Request = GetPositionIKRequest()
        #GSV_Request = GetStateValidityRequest()
        GPIK_Request.ik_request.group_name = 'arm_controller'
        GPIK_Request.ik_request.robot_state.joint_state = self.joints
        GPIK_Request.ik_request.avoid_collisions = True
        # this poses is related to the reference frame of gazebo
        # the pose is set as a radian value between 1.571 and -1.571
        GPIK_Request.ik_request.pose_stamped.pose.position.x = action[0]
        GPIK_Request.ik_request.pose_stamped.pose.position.y = action[1]
        GPIK_Request.ik_request.pose_stamped.pose.position.z = action[2]
        quaternion = quaternion_from_euler(0, 1.571, action[3], 'sxyz')
        GPIK_Request.ik_request.pose_stamped.pose.orientation.w = quaternion[0]
        GPIK_Request.ik_request.pose_stamped.pose.orientation.x = quaternion[1]
        GPIK_Request.ik_request.pose_stamped.pose.orientation.y = quaternion[2]
        GPIK_Request.ik_request.pose_stamped.pose.orientation.z = quaternion[3]
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

    # functions of execute trajectories
    def ee_traj_callback(self, request):

        self.pose_target.orientation.w = request.pose.orientation.w
        self.pose_target.orientation.x = request.pose.orientation.x
        self.pose_target.orientation.y = request.pose.orientation.y
        self.pose_target.orientation.z = request.pose.orientation.z
        self.pose_target.position.x = request.pose.position.x
        self.pose_target.position.y = request.pose.position.y
        self.pose_target.position.z = request.pose.position.z
        self.group.set_pose_target(self.pose_target)
        self.execute_trajectory()

        response = EeTrajResponse()
        response.success = True
        response.message = "Everything went OK"

        return response

    def joint_traj_callback(self, request):

        self.group_variable_values = self.group.get_current_joint_values()
        # rospy.spin()
        #print ("Group Vars:")
        #print (self.group_variable_values)
        #print ("Point:")
        #print (request.point.positions)
        self.group_variable_values[0] = request.point.positions[0]
        self.group_variable_values[1] = request.point.positions[1]
        self.group_variable_values[2] = request.point.positions[2]
        self.group_variable_values[3] = request.point.positions[3]
        self.group_variable_values[4] = request.point.positions[4]
        self.group_variable_values[5] = request.point.positions[5]
        # self.group_variable_values[6] = request.point.positions[6]
        self.group.set_joint_value_target(self.group_variable_values)
        self.execute_trajectory()

        response = JointTrajResponse()
        response.success = True
        response.message = "Everything went OK"

        return response

    def execute_trajectory(self):

        self.plan = self.group.plan()
        self.group.go(wait=True)

    def ee_pose_callback(self, request):

        gripper_pose = self.group.get_current_pose()
        #print (gripper_pose)

        gripper_pose_res = EePoseResponse()
        gripper_pose_res = gripper_pose

        return gripper_pose_res

    def ee_rpy_callback(self, request):

        gripper_rpy = self.group.get_current_rpy()
        #print (gripper_rpy)
        gripper_rpy_res = EeRpyResponse()
        gripper_rpy_res.r = gripper_rpy[0]
        gripper_rpy_res.y = gripper_rpy[1]
        gripper_rpy_res.p = gripper_rpy[2]

        return gripper_rpy_res


    # ParticularEnv methods
    # ---------------------------- in abb reach

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
