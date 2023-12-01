#! /usr/bin/env python
import sys
import rospy
import moveit_commander
import numpy as np
from openai_ros import robot_gazebo_env
from openai_ros.openai_ros_common import ROSLauncher
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA

class TiagoEnv(robot_gazebo_env.RobotGazeboEnv):
    """
    Initializes a new Tiago Steel environment
    """
    def __init__(self):
        rospy.logdebug("========= In Tiago Env")

        self.controllers_list = []
        self.robot_name_space = ""

        # Whether to reset controllers when a new episode starts
        reset_controls_bool = False
        
        # Parent class init
        super(TiagoEnv, self).__init__(controllers_list=self.controllers_list,
                                       robot_name_space=self.robot_name_space,
                                       reset_controls=reset_controls_bool,
                                       reset_world_or_sim="WORLD")
        
        self.gazebo.unpauseSim()
        self._init_moveit()
        self._init_rviz()
        self._check_all_systems_ready()

        self.gazebo.pauseSim()

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """

    # TiagoEnv virtual methods
    # ----------------------------

    def _init_moveit(self):
        moveit_commander.roscpp_initialize(sys.argv)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()

        # TODO: add other groups: 'head', 'gripper'
        self.gripper_group = moveit_commander.MoveGroupCommander('gripper')
        self.store_gripper_state()

        self.arm_group = moveit_commander.MoveGroupCommander('arm')
        self.arm_group.set_planning_time(0.1)
        self.store_arm_state()

        self.arm_workspace_low =  np.array([0.40, -0.25, 0.65])
        self.arm_workspace_high = np.array([0.75,  0.25, 0.75])

        rospy.sleep(2)
        p = PoseStamped()
        p.header.frame_id = robot.get_planning_frame()
        p.pose.position.x = 0.8
        p.pose.position.y = 0
        p.pose.position.z = 0.2

        scene.add_box('table', p, (1, 1, 0.4))

    def _init_rviz(self):
        self.marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=100, latch=True)

    # Methods that the TrainingEnvironment will need.
    # ----------------------------

    def visualize_points(self, x, y, z, ns=''):
        marker = Marker()
        marker.header.frame_id = '/base_link'
        marker.type = marker.SPHERE_LIST
        marker.action = marker.ADD
        marker.ns = ns
        marker.id = 0

        if ns == 'goal':
            marker.color.g = 1.0
        elif ns == 'action':
            marker.color.b = 1.0

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        if type(x) == list:
            for x, y, z in zip(x, y, z):
                marker.points.append(Point(x, y, z-0.3))
        else:
            marker.points.append(Point(x, y, z-0.3))

        marker.color.a = 0.5

        self.marker_publisher.publish(marker)

    def visualize_action(self, prev_x, prev_y, prev_z, x, y, z, valid):
        marker = Marker()
        marker.header.frame_id = '/base_link'
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.ns = 'action'
        marker.id = 0

        marker.scale.x = 0.01
        marker.scale.y = 0.02

        if valid:
            marker.color.a = 1
            marker.color.b = 1
        else:
            marker.color.a = 0.5

        marker.points.append(Point(prev_x, prev_y, prev_z-0.3))
        marker.points.append(Point(x, y, z-0.3))

        self.marker_publisher.publish(marker)

    def set_gripper_joints(self, l, r):
        self.gripper_group.set_joint_value_target([l, r])
        plan = self.gripper_group.go(wait=True)
        self.gripper_group.stop()
        self.store_gripper_state()
        return plan

    def set_arm_pose(self, x, y, z, roll, pitch, yaw):
        #[x, y, z] = np.clip([x, y, z], self.arm_workspace_low, self.arm_workspace_high)
        if not self.arm_pose_reachable(x, y, z):
            return False
        self.arm_group.set_pose_target([x, y, z, roll, pitch, yaw])

        plan = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        self.store_arm_state()
        return plan
    
    def shift_arm_pose(self, delta, dim):
        dim_list = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

        self.arm_group.shift_pose_target(value=delta, axis=dim_list.index(dim))
        plan = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        self.store_arm_state()
        return plan

    # Method must be called when sim is unpaused!
    def store_arm_state(self):
        pose = self.arm_group.get_current_pose().pose
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z
        roll, pitch, yaw = self.arm_group.get_current_rpy()

        self.stored_arm_state = [x, y, z, roll, pitch, yaw]

    def store_gripper_state(self):
        self.stored_gripper_state = self.gripper_group.get_current_joint_values()

    def arm_pose_reachable(self, x, y, z):
        return (([x, y, z] >= self.arm_workspace_low) & ([x, y, z] <= self.arm_workspace_high)).all()

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
