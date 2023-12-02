from pickle import FALSE
import numpy as np
import rospy
import time
import tf2_ros
from openai_ros import robot_gazebo_env
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2,JointState
from std_msgs.msg import Float32MultiArray
import sys
import copy
import rospy
from geometry_msgs.msg import Pose
from math import pi, tau, dist, fabs, cos
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from ur5e_gym.robot_movement import SmartGrasper
from cv_bridge import CvBridge 
from moveit_msgs.msg import PlanningScene

class Ur5eEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all ur5e environments.
    """

    def __init__(self):
        """
        Initializes a new ur5e environment.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        
        The Sensors: The sensors accesible are the ones considered usefull for AI learning.
        
        Sensor Topic List:
        * /camera/depth/image_raw
        * /camera/depth/points
        * /camera/rgb/image_raw
        * /laser_scan: Laser scan of the TCP
        * /iri_wam/iri_wam_controller/state, control_msgs/JointTrajectoryControllerState: Gives desired, actual and error.
        
        Actuators Topic List:
        * We publish int the action: /iri_wam/iri_wam_controller/follow_joint_trajectory/goal
        
        Args:
        """
        
        # Variables that we give through the constructor.
        # None in this case

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = ["gripper_controller","arm_controller","joint_state_controller"]
        rospy.loginfo("Starting Ur5e environment")
        # It doesnt use namespace
        self.bridge = CvBridge()
        self.robot_name_space = ""
        
        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(Ur5eEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=True, #change to false
                                            reset_world_or_sim="NO_RESET_SIM")
        #NO_RESET_SIM, reset controls false has won

       
        rospy.logdebug("Ur5eEnv unpause...")
        rospy.loginfo("Unpausing gazebo sim ...")
        self.gazebo.unpauseSim()
        rospy.loginfo("Checking systems ...")
        self._check_all_systems_ready()
        
        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self._camera_depth_image_raw_callback)
        rospy.Subscriber("/camera/depth/color/points", PointCloud2, self._camera_depth_points_callback)
        rospy.Subscriber("/camera/color/image_raw", Image, self._camera_rgb_image_raw_callback)
        rospy.Subscriber("/joint_states", JointState, self._joint_states_cb)
        rospy.Subscriber("/features/color", Float32MultiArray, self._color_featires_cb)
        rospy.Subscriber("/features/depth", Float32MultiArray, self._depth_featires_cb)
     
        
        self._setup_smart_grasper()
        self._setup_tf_listener()
        

        self.gazebo.pauseSim()
        
        rospy.loginfo("Finished Ur5e Environment ...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        
        self._check_all_sensors_ready()


        return True


    def _check_all_sensors_ready(self):
        
        # TODO: Here go the sensors like cameras and joint states
        self._check_camera_depth_image_raw_ready()
        self._check_camera_depth_points_ready()
        self._check_camera_rgb_image_raw_ready()
        self._check_joints_ready()
        self._check_tf()
        self._check_color_features()
        self._check_depth_features()
        
        rospy.loginfo("all sensors ready")
        
    def _check_joints_ready(self):

        joints_position = None 
        rospy.logdebug("Waiting for /joint_states to be READY...")
        while joints_position is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message("/joint_states",JointState, timeout=5.0)
                joints_position = msg
                rospy.logdebug("Current /joint_states READY=>")
                

            except:
                rospy.logerr("Current joints comunication not ready yet, retrying again")
               
        return joints_position 
    def _check_tf(self):

        tf = None 
        rospy.logdebug("Waiting for /tf to be READY...")
        while tf is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message("/tf",TFMessage, timeout=5.0)
                tf = msg
                rospy.logdebug("Current /tf READY=>")
                

            except:
                rospy.logerr("Current tf comunication not ready yet, retrying again")
               
        return tf
    def _check_color_features(self):

        color = None 
        rospy.logdebug("Waiting for /features/color to be READY...")
        while color is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message("/features/color",Float32MultiArray, timeout=5.0)
                color = msg
                rospy.logdebug("Current /features/color READY=>")
                

            except:
                rospy.logerr("Current /features/color comunication not ready yet, retrying again")
               
        return color

    def _check_depth_features(self):

        depth = None 
        rospy.logdebug("Waiting for /features/depth to be READY...")
        while depth is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message("/features/depth",Float32MultiArray, timeout=5.0)
                depth = msg
                rospy.logdebug("Current /features/depth READY=>")
                

            except:
                rospy.logerr("Current /features/depth comunication not ready yet, retrying again")
               
        return depth


    def _check_camera_depth_image_raw_ready(self):
        camera_depth_image_raw = None
        rospy.logdebug("Waiting for /camera/depth/image_rect_raw to be READY...")
        while camera_depth_image_raw is None and not rospy.is_shutdown():
            try:
                camera_depth_image_raw = rospy.wait_for_message("/camera/depth/image_rect_raw", Image, timeout=5.0)
                rospy.logdebug("Current /camera/depth/image_raw READY=>")

            except:
                rospy.logerr("Current /camera/depth/image_raw not ready yet, retrying for getting camera_depth_image_raw")
        return camera_depth_image_raw
        
        
    def _check_camera_depth_points_ready(self):
        camera_depth_points = None
        rospy.logdebug("Waiting for /camera/depth/color/points to be READY...")
        while camera_depth_points is None and not rospy.is_shutdown():
            try:
                camera_depth_points = rospy.wait_for_message("/camera/depth/color/points", PointCloud2, timeout=10.0)
                rospy.logdebug("Current /camera/depth/points READY=>")

            except:
                rospy.logerr("Current /camera/depth/color/points not ready yet, retrying for getting camera_depth_points")
        return camera_depth_points
        
        
    def _check_camera_rgb_image_raw_ready(self):
        camera_rgb_image_raw = None
        rospy.logdebug("Waiting for /camera/rgb/image_raw to be READY...")
        while camera_rgb_image_raw is None and not rospy.is_shutdown():
            try:
                camera_rgb_image_raw = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=5.0)
                rospy.logdebug("Current /camera/color/image_raw READY=>")

            except:
                rospy.logerr("Current /camera/color/image_raw not ready yet, retrying for getting camera_rgb_image_raw")
        return camera_rgb_image_raw

    def _check_planning_scene_ready(self):
        planning_scene = None
        rospy.logdebug("Waiting for /planning_scene to be READY...")
        while planning_scene is None and not rospy.is_shutdown():
            try:
                planning_scene = rospy.wait_for_message('/planning_scene', PlanningScene, timeout=1.0)
                rospy.logdebug("Current /planning_scene READY=>")

            except:
                rospy.logerr("Current /planning_scene not ready yet, retrying for getting planning_scene")
        return planning_scene
        
      
        

        
    def _camera_depth_image_raw_callback(self, data):
        
        self.camera_depth_image_raw = self.bridge.imgmsg_to_cv2(data,desired_encoding='passthrough')
        
    def _camera_depth_points_callback(self, data):
        self.camera_depth_points = data
    
    def _color_featires_cb(self, data):
        self.color_features = data.data

    def _depth_featires_cb(self, data):
        self.depth_features = data.data        
        
    def _camera_rgb_image_raw_callback(self, data):        
        self.camera_rgb_image_raw = self.bridge.imgmsg_to_cv2(data,desired_encoding="rgb8")        
     
    
    def _setup_tf_listener(self):
        """
        Set ups the TF listener for getting the transforms you ask for.
        """
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)


        
    def _setup_smart_grasper(self):
        """
        Setup of the movement system.
        :return:
        """
        self.movement_system = SmartGrasper()
    def _get_tip_pose(self,reference="base_link",tip="tcp"):
        """
        Return the pose of the tip frame.
        :return:
        """
        tip_pose = Pose()
        is_tf_ready = self.tf_buffer.can_transform(reference,tip,rospy.Time(),timeout=rospy.Duration(5.0))
        if  is_tf_ready:
            trans = self.tf_buffer.lookup_transform(reference,tip, rospy.Time())
            tip_pose.position.x = trans.transform.translation.x
            tip_pose.position.y = trans.transform.translation.y           
            tip_pose.position.z = trans.transform.translation.z
            tip_pose.orientation.x = trans.transform.rotation.x
            tip_pose.orientation.y = trans.transform.rotation.y 
            tip_pose.orientation.z = trans.transform.rotation.z 
            tip_pose.orientation.w = trans.transform.rotation.w  
        else :
            try:
               tip_pose =  self.movement_system.get_tip_pose()
            except:
                rospy.logerr("an error ocurred getting the tip pose")
        return tip_pose

    def _get_fingers_pose(self):
        """
        Return the pose of the left and right finger in that order.
        :return:
        """
        left_finger_pose = Pose()
        right_finger_pose = Pose()
        try:
            is_tf_ready = self.tf_buffer.can_transform("base_link","right_finger_v1_1",rospy.Time(),timeout=rospy.Duration(5.0))
            trans_rf = self.tf_buffer.lookup_transform("base_link","right_finger_v1_1", rospy.Time())
            right_finger_pose.position.x = trans_rf.transform.translation.x
            right_finger_pose.position.y = trans_rf.transform.translation.y
            right_finger_pose.position.z = trans_rf.transform.translation.z

            trans_lf = self.tf_buffer.lookup_transform("base_link","left_finger_v1_1", rospy.Time())
            left_finger_pose.position.x = trans_lf.transform.translation.x
            left_finger_pose.position.y = trans_lf.transform.translation.y
            left_finger_pose.position.z = trans_lf.transform.translation.z
        except:
            rospy.logerr("an error ocurred getting the fingers pose")
        return left_finger_pose,right_finger_pose

    

         
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
    
        
      
    def get_fingers_colision(self, object_collision_name):
        """
        Returns the collision of the three fingers
        object_collision_name: Here yo ustate the name of the model to check collision
        with fingers.
        Objects in sim: mytable,action_table,mycyl
        """
        self.gazebo.unpauseSim()
        self.movement_system .enable_fingers_collisions(True)
        planning_scene = self._check_planning_scene_ready()
        self.gazebo.pauseSim()
        
        objects_scene = planning_scene.allowed_collision_matrix.entry_names
        colissions_matrix = planning_scene.allowed_collision_matrix.entry_values
        
        # We look for the Ball object model name in the objects sceen list and get the index:
        object_collision_name_index = objects_scene.index(object_collision_name)
        
        Finger_Links_Names = [ "left_finger_v1_1","right_finger_v1_1","gripper_base_link"]
                    
                             
        # We get all the index of the model links that are part of the fingers
        # We separate by finguer to afterwards be easy to detect that there is contact in all of the finguers
        finger1_indices = [i for i, var in enumerate(Finger_Links_Names) if "left" in var]
        finger2_indices = [i for i, var in enumerate(Finger_Links_Names) if "right" in var]
                
        # Now we search in the entry_value corresponding to the object to check the collision
        # With all the rest of objects.
        object_collision_array = colissions_matrix[object_collision_name_index].enabled
        

        f1_collision = False
        for finger_index in finger1_indices:
            if object_collision_array[finger_index]:
                f1_collision = True
                break
        

        f2_collision = False
        for finger_index in finger2_indices:
            if object_collision_array[finger_index]:
                f2_collision = True
                break
            

                    
        finger_collision_dict = {   
                                    "f":f1_collision,
                                    "f2":f2_collision,
                                }
        
        return finger_collision_dict

    def get_joint_states(self):
        self.joint_states = self.movement_system.__last_joint_state
        return self.joint_states

    def _joint_states_cb(self,data):
        self.joint_states = data
        
   