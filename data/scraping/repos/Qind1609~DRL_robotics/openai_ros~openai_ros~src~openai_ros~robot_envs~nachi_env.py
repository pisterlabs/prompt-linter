import numpy as np
import rospy
import copy
from std_msgs.msg import Float64
from gazebo_msgs.srv import GetWorldProperties, GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import JointState
from openai_ros.robot_gazebo_env import RobotGazeboEnv
from openai_ros.openai_ros_common import ROSLauncher
import sys
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
import shape_msgs.msg


class NachiEnv(RobotGazeboEnv):
    def __init__(self, ros_ws_abspath):

        # We launch the ROSlaunch that spawns the robot into the world
        ROSLauncher(
            rospackage_name="openai_hac",
            launch_file_name="put_nachi_to_world.launch",
            ros_ws_abspath=ros_ws_abspath,
        )

        # "arm_controller", "joint_state_controller"
        self.controllers_list = ["mz07_controller", "joint_state_controller"]

        self.robot_name_space = "/mz07"
        self.reset_controls = False

        self.obj_positions = Obj_Pos()

        super(NachiEnv, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=self.reset_controls,
            start_init_physics_parameters=True,
            reset_world_or_sim="WORLD",
        )

        # We Start all the ROS related Subscribers and publishers

        self.JOINT_STATES_SUBSCRIBER = "/mz07/joint_states"
        self.joint_names = [
            "J1",
            "J2",
            "J3",
            "J4",
            "J5",
            "J6",
        ]

        self.gazebo.unpauseSim()
        self._check_all_systems_ready()

        self.joint_states_sub = rospy.Subscriber(
            self.JOINT_STATES_SUBSCRIBER, JointState, self.joints_callback
        )
        self.joints = JointState()

        # Start Move robot, that checks all systems are ready
        self.move_robot = RobotMove()
        # Wait until Robot goes to the init pose
        self.wait_robot_ready()

        # We pause until the next step
        self.gazebo.pauseSim()

    # RobotGazeboEnv virtual methods
    # ----------------------------
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True

    def _check_all_sensors_ready(self):
        self._check_joint_states_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_joint_states_ready(self):
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message(
                    self.JOINT_STATES_SUBSCRIBER, JointState, timeout=1.0
                )
                rospy.logdebug(
                    "Current "
                    + str(self.JOINT_STATES_SUBSCRIBER)
                    + " READY=>"
                    + str(self.joints)
                )
            except:
                rospy.logerr(
                    "Current "
                    + str(self.JOINT_STATES_SUBSCRIBER)
                    + " not ready yet, retrying...."
                )
        return self.joints

    def joints_callback(self, data):
        self.joints = data

    def get_joints(self):
        return self.joints

    def get_joint_names(self):
        return self.joints.name

    def set_trajectory_ee(self, action):
        """
        Sets the Pose of the EndEffector based on the action variable.
        The action variable contains the position and orientation of the EndEffector.
        See create_action
        """
        # Set up a trajectory message to publish.
        ee_target = geometry_msgs.msg.Pose()
        ee_target.position.x = action[0]
        ee_target.position.y = action[1]
        ee_target.position.z = action[2]

        # orientation in quarternion
        ee_target.orientation.x = action[3]
        ee_target.orientation.y = action[4]
        ee_target.orientation.z = action[5]
        ee_target.orientation.w = action[6]

        # clip the action values to forcefully have them fall into a safe range for the execution.
        ee_target = self.trajectory_processing(ee_target)
        rospy.logwarn("============== Action: {}".format(ee_target))
        result = self.move_robot.ee_traj(ee_target)
        return result

    def set_trajectory_joints(self, initial_qpos):

        positions_array = [None] * 6
        positions_array[0] = initial_qpos["J1"]
        positions_array[1] = initial_qpos["J2"]
        positions_array[2] = initial_qpos["J3"]
        positions_array[3] = initial_qpos["J4"]
        positions_array[4] = initial_qpos["J5"]
        positions_array[5] = initial_qpos["J6"]

        result = self.move_robot.joint_traj(positions_array)

        return result

    def trajectory_processing(self, ee_target):
        """
        We clip the values within a defined range not to have an abortive execution of trajectory
        the available init region such as max or min of xyz must be specified in task env!!

        Example:
            self._x_min = 0.25
            self._x_max = 1.5
            self._y_min = -0.3
            self._y_max = 0.3
            self._z_min = 0.5
            self._z_max = 1.5
        """

        _pose = self.get_ee_pose()
        x = _pose.pose.position.x
        y = _pose.pose.position.y
        z = _pose.pose.position.z

        ee_target.position.x = np.clip(
            x + ee_target.position.x, self.position_ee_x_min, self.position_ee_x_max
        )
        ee_target.position.y = np.clip(
            y + ee_target.position.y, self.position_ee_y_min, self.position_ee_y_max
        )
        ee_target.position.z = np.clip(
            z + ee_target.position.z, self.position_ee_z_min, self.position_ee_z_max
        )

        return ee_target

    def create_joints_dict(self, joints_positions):
        """
        Based on the Order of the positions, they will be assigned to its joint name
        names_in_order:
          joint0: 0.0
          joint1: 0.0
          joint2: 0.0
          joint3: -1.5
          joint4: 0.0
          joint5: 1.5

        """

        assert len(joints_positions) == len(
            self.joint_names
        ), "Wrong number of joints, there should be " + str(len(self.join_names))
        joints_dict = dict(zip(self.joint_names, joints_positions))

        return joints_dict

    def get_ee_pose(self):

        """
        Returns geometry_msgs/PoseStamped
            std_msgs/Header header
              uint32 seq
              time stamp
              string frame_id
            geometry_msgs/Pose pose
              geometry_msgs/Point position
                float64 x
                float64 y
                float64 z
              geometry_msgs/Quaternion orientation
                float64 x
                float64 y
                float64 z
                float64 w
        """

        self.gazebo.unpauseSim()
        gripper_pose = self.move_robot.ee_pose()
        print(gripper_pose)
        # self.gazebo.pauseSim()
        return gripper_pose

    def get_ee_rpy(self):
        self.gazebo.unpauseSim()
        gripper_rpy = self.move_robot.ee_rpy()
        # self.gazebo.pauseSim()
        return gripper_rpy

    def wait_robot_ready(self):
        print("WAITING FOR RVIZ")
        rospy.sleep(5)

        print("WAITING...DONE")

    # ParticularEnv methods
    # ----------------------------

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _set_init_pose(self):
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given."""
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation."""
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given."""
        raise NotImplementedError()
    def _set_goal(self, x, y, z, rx, ry, rz, w):
        return NotImplementedError()

class Obj_Pos(object):

    """
    This object maintains the pose and rotation of the cube in a simulation
    through Gazebo Service

    """

    def __init__(self):

        rospy.wait_for_service("/gazebo/get_model_state")
        rospy.wait_for_service("/gazebo/set_model_state")

        self.time = 0
        self.model_names = ["cube"]
        self.get_model_state = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )
        self.reset_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    def reset_position(self, x, y, z, model_name="cube"):
        """
        Randomly initialise the position of the cube on the table
        """
        # TODO: Add random orientation
        state_msg = ModelState()
        # target object to relocate
        state_msg.model_name = model_name

        # set random numbers by taking a size of the cube and the table into consideration
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = z
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 1.0

        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            self.reset_state(state_msg)
        except rospy.ServiceException as e:
            rospy.logdebug("Service call failed: %s" % e)

    def get_states(self, model_name="cube"):
        """
        Returns the ndarray of pose&rotation of the cube
        """
        for modelname in self.model_names:
            if modelname == model_name:
                data = self.get_model_state(modelname, "world")  # gazebo service client
                return np.array(
                    [
                        data.pose.position.x,
                        data.pose.position.y,
                        data.pose.position.z,
                        data.pose.orientation.x,
                        data.pose.orientation.y,
                        data.pose.orientation.z,
                        data.pose.orientation.w,
                    ]
                )


class RobotMove(object):
    def __init__(self):
        joint_state_topic = ["/joint_states:=/mz07/joint_states"]
        moveit_commander.roscpp_initialize(joint_state_topic)
        moveit_commander.roscpp_initialize(sys.argv)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("arm")
        self.group.set_pose_reference_frame("base_link")

        self.group.set_planning_time(0.04)
        self.group.set_goal_position_tolerance(1e-4)
        self.group.set_goal_orientation_tolerance(1e-4)
        self.group.set_start_state_to_current_state()

    def ee_traj(self, pose):

        self.group.set_pose_target(pose)
        # plan, _ = self.plan_cartesian_path(pose)
        # result = self.execute_plan(plan)
        result = self.execute_trajectory_pose()
        return result

    def joint_traj(self, positions_array):

        self.group_variable_values = self.group.get_current_joint_values()
        self.group_variable_values[0] = positions_array[0]
        self.group_variable_values[1] = positions_array[1]
        self.group_variable_values[2] = positions_array[2]
        self.group_variable_values[3] = positions_array[3]
        self.group_variable_values[4] = positions_array[4]
        self.group_variable_values[5] = positions_array[5]
        self.group.set_joint_value_target(self.group_variable_values)
        result = self.execute_trajectory_joint()
        return result

    def plan_cartesian_path(self, pose_target):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.group

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = move_group.compute_cartesian_path(
            pose_target, 0.01, 0.0  # waypoints to follow  # eef_step
        )  # jump_threshold

        # Note: We are just planning, not asking move_group to actually move the robot yet:
        return plan, fraction

    def execute_plan(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.group

        ## BEGIN_SUB_TUTORIAL execute_plan
        ##
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        result = move_group.execute(plan, wait=True)
        return result

    def execute_trajectory_joint(self):
        """
        Assuming that the trajecties has been set to the self objects appropriately
        Make a plan to the destination in Homogeneous Space(x,y,z,yaw,pitch,roll)
        and returns the result of execution
        """

        self.plan = self.group.plan()
        result = self.group.go(self.group_variable_values, wait=True)

        self.group.stop()
        return result

    def execute_trajectory_pose(self):
        """
        Assuming that the trajecties has been set to the self objects appropriately
        Make a plan to the destination in Homogeneous Space(x,y,z,yaw,pitch,roll)
        and returns the result of execution
        """

        self.plan = self.group.plan()
        result = self.group.go(wait=True)

        self.group.stop()
        self.group.clear_pose_targets()

        return result

    def ee_pose(self):
        gripper_pose = self.group.get_current_pose()
        return gripper_pose

    def ee_rpy(self):
        gripper_rpy = self.group.get_current_rpy()
        return gripper_rpy
