import numpy as np
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64, Header
from sensor_msgs.msg import JointState, LaserScan, Image, PointCloud2, PointField
from actionlib import GoalStatusArray
from geometry_msgs.msg import Twist, Pose, PoseStamped, Transform, TransformStamped
from openai_ros.openai_ros_common import ROSLauncher

import actionlib
import sys
import moveit_commander
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from controller_manager_msgs.srv import SwitchController
from gazebo_msgs.srv import SetModelConfiguration, GetModelState, GetLinkState
from gazebo_msgs.srv import SetLinkProperties, SetLinkPropertiesRequest
import time
import open3d
from scipy.spatial.transform import Rotation
import sensor_msgs.point_cloud2 as pc2
import tf2_sensor_msgs
import tf2_ros


class IpaKIFZEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all view planning environments.
    """
    def __init__(self):
        rospy.logdebug("Start IpaKIFZEnv INIT...")

        if self.sensor_only:
            self.controllers_list = []
            reset_controls = False
        else:
            self.controllers_list = [
                "arm_controller", "joint_state_controller"
            ]
            reset_controls = True

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(IpaKIFZEnv,
              self).__init__(controllers_list=self.controllers_list,
                             robot_name_space="",
                             reset_controls=reset_controls,
                             start_init_physics_parameters=False,
                             reset_world_or_sim="WORLD")

        self._init_env()

        # self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        self._check_all_systems_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/point_cloud", PointCloud2,
                         self._point_cloud_callback)

        # self.gazebo.pauseSim()

        rospy.logdebug("Finished IpaKIFZEnv INIT...")

    def _init_env(self):
        if not self.sensor_only:
            # Init MoveIt
            moveit_commander.roscpp_initialize(sys.argv)
            self.robot = moveit_commander.RobotCommander()
            # self.move_group = moveit_commander.MoveGroupCommander("welding_endeffector")
            self.move_group = moveit_commander.MoveGroupCommander(
                "sensor_endeffector")
            self.scene = moveit_commander.PlanningSceneInterface()

            # Init TF
            self.tfBuffer = tf2_ros.Buffer()
            self.listener = tf2_ros.TransformListener(self.tfBuffer)
        else:
            # Service to set sensor position
            self.set_model_configuration = rospy.ServiceProxy(
                'gazebo/set_model_configuration', SetModelConfiguration)
            self.get_model_state = rospy.ServiceProxy('gazebo/get_model_state',
                                                      GetModelState)
            self.get_link_state = rospy.ServiceProxy('gazebo/get_link_state',
                                                     GetLinkState)

        self.planning_time = 0
        self.planning_num = 0
        self.measurement_time = 0
        self.measurement_num = 0

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_point_cloud_ready()
        return True

    def _check_point_cloud_ready(self):
        self.point_cloud = None
        rospy.logdebug("Waiting for /point_cloud to be READY...")
        while self.point_cloud is None and not rospy.is_shutdown():
            try:
                self.point_cloud = rospy.wait_for_message("/point_cloud",
                                                          PointCloud2,
                                                          timeout=5.0)
                rospy.logdebug("Current /point_cloud READY=>")
            except:
                rospy.logerr(
                    "Current /point_cloud not ready yet, retrying for getting point_cloud"
                )
        return self.point_cloud

    def _point_cloud_callback(self, data):
        # self.point_cloud = data
        pass

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
    def plan_pose(self, pose):
        """
        It will move the robot to the base if a movement has been found
        :param pose: 6d pose
        :return:
        """
        start_time = time.time()

        # Option 1 Plan and Execute with MoveIt
        if not self.sensor_only:
            self.move_group.set_pose_target(pose)
            success = self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()

        # Option 2 Set Model Configuration, i.e. joint states
        else:
            # Save current pose and set the fake joints
            joint_names = [
                'joint_trans_x', 'joint_trans_y', 'joint_trans_z',
                'joint_rot_x', 'joint_rot_y', 'joint_rot_z'
            ]
            pose_quaternion = [
                pose.orientation.x, pose.orientation.y, pose.orientation.z,
                pose.orientation.w
            ]
            pose_euler = Rotation.from_quat(pose_quaternion).as_euler('XYZ')
            joint_positions = [
                pose.position.x, pose.position.y, pose.position.z,
                pose_euler[0], pose_euler[1], pose_euler[2]
            ]

            # Set the Sensor position in Gazebo
            set_model_configuration_resp = self.set_model_configuration(
                'sensor', 'sensor_description', joint_names, joint_positions)
            success = set_model_configuration_resp.success

            # get_model_state_resp = self.get_model_state('sensor', '')
            time.sleep(0.05)  #TODO: Tune physics parameter to avoid shifting!

        end_time = time.time()
        self.planning_time += end_time - start_time
        self.planning_num += 1
        # rospy.logdebug("Sensor Positioning Time: "+str(self.planning_time/self.planning_num))

        return success

    def get_current_pose(self):
        if not self.sensor_only:
            return self.move_group.get_current_pose().pose
        else:
            link_state_resp = self.get_link_state("link_6", "sensor")
            current_sensor_pose = link_state_resp.link_state.pose
            return current_sensor_pose

    def get_open3d_point_cloud(self):
        start_time = time.time()

        # Wait on sensor measurement
        self.point_cloud = self._check_point_cloud_ready()

        # Transform Point Cloud to world coordinate system
        if not self.sensor_only:
            trafo_sensor_world = self.tfBuffer.lookup_transform(
                'world', 'ensenso_sim_link', rospy.Time(), rospy.Duration(1.0))
        else:
            trafo_sensor_world = TransformStamped()
            current_sensor_pose = self.get_current_pose()
            trafo_sensor_world.transform.translation.x = current_sensor_pose.position.x
            trafo_sensor_world.transform.translation.y = current_sensor_pose.position.y
            trafo_sensor_world.transform.translation.z = current_sensor_pose.position.z
            trafo_sensor_world.transform.rotation.x = current_sensor_pose.orientation.x
            trafo_sensor_world.transform.rotation.y = current_sensor_pose.orientation.y
            trafo_sensor_world.transform.rotation.z = current_sensor_pose.orientation.z
            trafo_sensor_world.transform.rotation.w = current_sensor_pose.orientation.w

        self.point_cloud = tf2_sensor_msgs.do_transform_cloud(
            self.point_cloud, trafo_sensor_world)
        self.point_cloud.header.frame_id = "world"

        # Convert ROS sensor_msgs::PointCloud2 to Open3d format
        open3d_cloud = self.convertROStoOpen3d(self.point_cloud)

        end_time = time.time()
        self.measurement_time += end_time - start_time
        self.measurement_num += 1
        # rospy.logdebug("Sensor Positioning Time: "+str(self.measurement_time/self.measurement_num))

        return open3d_cloud

    def convertROStoOpen3d(self, point_cloud):
        field_names = ['x', 'y', 'z']
        cloud_data = list(
            pc2.read_points(point_cloud,
                            skip_nans=True,
                            field_names=field_names))
        open3d_cloud = open3d.geometry.PointCloud()
        if len(cloud_data) > 0:
            xyz = [(x, y, z) for x, y, z in cloud_data]
            open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))

        return open3d_cloud

    def convertOpen3dtoROS(self, open3d_cloud, frame_id='world'):
        fields_xyz = [
            PointField(name='x',
                       offset=0,
                       datatype=PointField.FLOAT32,
                       count=1),
            PointField(name='y',
                       offset=4,
                       datatype=PointField.FLOAT32,
                       count=1),
            PointField(name='z',
                       offset=8,
                       datatype=PointField.FLOAT32,
                       count=1),
        ]

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        cloud_out = pc2.create_cloud(header, fields_xyz,
                                     np.asarray(open3d_cloud.points))
        return cloud_out
