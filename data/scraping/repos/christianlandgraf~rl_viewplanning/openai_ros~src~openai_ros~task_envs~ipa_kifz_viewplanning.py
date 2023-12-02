import rospy
import rospkg
from math import ceil, sqrt
import numpy as np
import pandas as pd
from gym import spaces
from openai_ros.robot_envs import ipa_kifz_env
from gym.envs.registration import register
from geometry_msgs.msg import Point, Pose, PoseStamped
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os
import time
import itertools

import tf_conversions
from scipy.spatial.transform import Rotation

import sys
sys.path.append(".")
import open3d
import stl
import sensor_msgs.point_cloud2 as pc2

from ipa_kifz_viewplanning.srv import GetAreaGain, GetAreaGainRequest, GetAreaGainResponse


class IpaKIFZViewplanningEnv(ipa_kifz_env.IpaKIFZEnv):
    def __init__(self):
        """
        This Task Env is designed for learning the best sensor measurement poses.
        """

        # Load Params from the desired Yaml file
        self.algorithm = rospy.get_param('/algorithm')
        if self.algorithm == "viewpoint_sampling":
            yaml_file_name = "ipa_kifz_viewpoint_sampling.yaml"
        elif self.algorithm == "generate_dataset":
            yaml_file_name = "ipa_kifz_dataset_generation.yaml"
        else:
            yaml_file_name = "ipa_kifz_viewplanning.yaml"
        LoadYamlFileParamsTest(
            rospackage_name="openai_ros",
            rel_path_from_package_to_file="src/openai_ros/task_envs/config",
            yaml_file_name=yaml_file_name)

        # Whether to simulate the sensor only or the whole robot
        self.sensor_only = rospy.get_param('/sensor_only')

        # Workpiece Poses
        self.workpiece_name = rospy.get_param(
            '/ipa_kifz_viewplanning/workpiece_name')
        self.workpiece_pose = rospy.get_param(
            '/ipa_kifz_viewplanning/workpiece_pose',
            default=[0, 0, 0, 0, 0, 0, 1])

        # Number of desired poses and coverage per episode
        self.desired_steps = rospy.get_param(
            '/ipa_kifz_viewplanning/desired_steps', default=10)
        self.desired_coverage = rospy.get_param(
            '/ipa_kifz_viewplanning/desired_coverage', default=0.95)

        # Whether to return reward at end of episode or iteration
        self.use_cumulated_reward = rospy.get_param(
            '/ipa_kifz_viewplanning/use_cumulated_reward')
        if self.use_cumulated_reward:
            self.actions_per_step = self.desired_steps
            self.desired_steps = 1
        else:
            self.actions_per_step = 1

        # (Re-)initialize the robot environment
        super(IpaKIFZViewplanningEnv, self).__init__()

        # Add the workpiece to the environment
        self.workpiece_handler()

        # Services and Publishers
        self.get_area_gain = rospy.ServiceProxy(
            'point_cloud_handler/get_area_gain', GetAreaGain)
        self.log_pose_pub = rospy.Publisher('/openai/episode_poses',
                                            Pose,
                                            queue_size=10)

        # Initial parameters defining the grid extend and location
        self.init_pos_z = rospy.get_param('/ipa_kifz_viewplanning/init_pos_z',
                                          default=0)
        self.init_rot_qx = rospy.get_param(
            '/ipa_kifz_viewplanning/init_rot_qx', default=0)
        self.init_rot_qy = rospy.get_param(
            '/ipa_kifz_viewplanning/init_rot_qy', default=0)
        self.init_rot_qz = rospy.get_param(
            '/ipa_kifz_viewplanning/init_rot_qz', default=0)
        self.init_rot_qw = rospy.get_param(
            '/ipa_kifz_viewplanning/init_rot_qw', default=1)

        # Get action space limits
        if rospy.has_param('/ipa_kifz_viewplanning/min_range_x'):
            self.min_range_x = rospy.get_param(
                '/ipa_kifz_viewplanning/min_range_x')
            self.min_range_y = rospy.get_param(
                '/ipa_kifz_viewplanning/min_range_y')
            self.min_range_z = rospy.get_param(
                '/ipa_kifz_viewplanning/min_range_z')
            self.max_range_x = rospy.get_param(
                '/ipa_kifz_viewplanning/max_range_x')
            self.max_range_y = rospy.get_param(
                '/ipa_kifz_viewplanning/max_range_y')
            self.max_range_z = rospy.get_param(
                '/ipa_kifz_viewplanning/max_range_z')
        else:
            self.set_grid_limits()

        # For a discretized action space
        self.is_discretized = rospy.get_param(
            '/ipa_kifz_viewplanning/is_discretized', default=False)
        if self.is_discretized:
            # Load grid parameters
            self.use_grid = rospy.get_param('/ipa_kifz_viewplanning/use_grid',
                                            default=False)
            if self.use_grid:
                self.triangle_grid = rospy.get_param(
                    '/ipa_kifz_viewplanning/triangle_grid')
                self.steps_yaw = rospy.get_param(
                    '/ipa_kifz_viewplanning/grid_steps_yaw')
                self.step_size_x = rospy.get_param(
                    '/ipa_kifz_viewplanning/grid_step_size_x')
                self.step_size_y = rospy.get_param(
                    '/ipa_kifz_viewplanning/grid_step_size_y')
                self.step_size_z = rospy.get_param(
                    '/ipa_kifz_viewplanning/grid_step_size_z')
        else:
            if self.algorithm == "qlearn":
                rospy.logerr(
                    "Q Learning cannot be appied to continuous spaces.")
                sys.exit()
            pass

        # setting up action space for training

        # Whether to simultanously test
        self.test_mode = rospy.get_param('/ipa_kifz_viewplanning/test_mode',
                                         default=False)

        # For Q Learning -> Discretize the action space (positions the robot can go to) as list of tuples!
        if self.is_discretized:
            # Use a regular x-y-z grid
            if self.use_grid:
                discretized_actions = self.get_grid()
                self.discretized_action_space = discretized_actions
            else:
                viewpoint_list = self.get_viewpoints()
                self.discretized_action_space = viewpoint_list

            if self.use_cumulated_reward:
                # Build all combinations of viewpoints and save them as individual lists
                self.discretized_action_space = list(
                    itertools.combinations(self.discretized_action_space,
                                           self.desired_poses))
                self.discretized_action_space = [
                    list(i) for i in self.discretized_action_space
                ]
            else:
                # Build arrays of one element to be compatible with the upper case
                self.discretized_action_space = [
                    [i] for i in self.discretized_action_space
                ]

            # Set action space to discrete number of actions
            self.action_space = spaces.Discrete(
                len(self.discretized_action_space))
        else:
            # Set continuous box action space
            # Consists of poses of form (x,y) (restricted to plane to reduce complexity)
            lower_bound = [0] * self.actions_per_step * 2
            upper_bound = [0] * self.actions_per_step * 2
            for i in range(self.actions_per_step):
                lower_bound[i * 2] = self.min_range_x
                lower_bound[i * 2 + 1] = self.min_range_y
                upper_bound[i * 2] = self.max_range_x
                upper_bound[i * 2 + 1] = self.max_range_y

            self.action_space = spaces.Box(low=np.array(lower_bound),
                                           high=np.array(upper_bound))

            # self.action_space = spaces.Box(low=np.array([self.min_range_x, self.min_range_y, self.min_range_z, 0, 210, 0]),
            #                             high=np.array([self.max_range_x, self.max_range_y, self.max_range_z, 360, 330, 360]))

        # Setting up Observation Space for training
        # Currently, the observation is the sensor position in x,y and z coordinates,
        # its quaternions consisting of four values between 0 and 1, and the area gain
        self.observation_space = spaces.Box(low=np.array([
            self.min_range_x, self.min_range_y, self.min_range_z, 0, 0, 0, 0, 0
        ]),
                                            high=np.array([
                                                self.max_range_x,
                                                self.max_range_y,
                                                self.max_range_z, 1, 1, 1, 1, 1
                                            ]))

        self.episode_steps = 1

        self.pc_handling_time = 0
        self.pc_handling_num = 0

        # Initializing variables for testing and debugging
        if self.test_mode:
            if self.is_discretized:
                self.area_gain_control = [0] * len(
                    self.discretized_action_space)
            sys.exit()

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        # Reset number of steps
        self.episode_steps = 1

        # Number of performed steps in total
        self.cumulated_steps = -1

        # Cumulated Point Cloud
        self.cumulated_point_cloud = None

        # All Actions which have been chosen in this episode
        self.cumulated_actions = []

        # View poses per episode
        self.current_poses = []

        # Cumulated area gain per episode
        self.cumulated_area_gain = 0

        return True

    def _set_init_pose(self):
        pass

    def _set_action(self, action):
        """
        Perform action, i.e. go to the pose
        :param action: The next pose of the robot
        """
        if self.is_discretized:
            rospy.logwarn("Start Set Action ==>" + str(np.around(action, 2)))
        else:
            rospy.logwarn("Start Set Action ==>" + str(action))

        next_pose = Pose()

        # In case of using a cumulated reward approach, reset area gain
        if self.use_cumulated_reward:
            self.area_gain = 0
            self.current_poses = [None] * self.actions_per_step

        for i in range(self.actions_per_step):
            # Option 1: RL Training, set next Action, i.e. next pose
            if self.is_discretized:
                next_pose.position.x = self.discretized_action_space[action][
                    i][0]
                next_pose.position.y = self.discretized_action_space[action][
                    i][1]
                next_pose.position.z = self.discretized_action_space[action][
                    i][2]

                next_pose.orientation.x = self.discretized_action_space[
                    action][i][3]
                next_pose.orientation.y = self.discretized_action_space[
                    action][i][4]
                next_pose.orientation.z = self.discretized_action_space[
                    action][i][5]
                next_pose.orientation.w = self.discretized_action_space[
                    action][i][6]

                # Option 2: Debugging, Go through the complete action space for debugging purposes
                if self.test_mode:
                    next_pose.position.x = self.discretized_action_space[
                        self.episode_steps - 1][i][0]
                    next_pose.position.y = self.discretized_action_space[
                        self.episode_steps - 1][i][1]

            else:
                # Option 3: Random sampling, for dataset generation
                if self.algorithm == "viewpoint_sampling" or self.algorithm == "dataset_generation":
                    next_pose.position.x = np.random.uniform(
                        self.min_range_x, self.max_range_x)
                    next_pose.position.y = np.random.uniform(
                        self.min_range_y, self.max_range_y)
                    next_pose.position.z = np.random.uniform(
                        self.min_range_z, self.max_range_z)
                    rotx = np.random.uniform(0, 360)
                    roty = np.random.uniform(210, 330)
                    rotz = np.random.uniform(0, 360)
                    quat = Rotation.from_euler('XYZ', [rotx, roty, rotz],
                                               degrees=True).as_quat()
                    next_pose.orientation.x = quat[0]
                    next_pose.orientation.y = quat[1]
                    next_pose.orientation.z = quat[2]
                    next_pose.orientation.w = quat[3]
                else:
                    # Option 4 Approach given pose from action (currently only x,y coordinates )
                    next_pose.position.x = action[i * 2]
                    next_pose.position.y = action[i * 2 + 1]
                    next_pose.position.z = self.init_pos_z  #action[2]

                    # quat = Rotation.from_euler('XYZ', [action[3], action[4], action[5]], degrees=True).as_quat()
                    next_pose.orientation.x = self.init_rot_qx
                    next_pose.orientation.y = self.init_rot_qy
                    next_pose.orientation.z = self.init_rot_qz
                    next_pose.orientation.w = self.init_rot_qw

            # Go to pose
            success = self.plan_pose(next_pose)

            # Publish pose for logger
            self.log_pose_pub.publish(next_pose)

            # If multiple poses are evaluated, immediately update the area gain
            if success and self.use_cumulated_reward:
                self.area_gain += self.evaluate_measurement()
                self.current_poses[i] = self.get_current_pose()

            # If no robot plan is found, go to default pose (reward = zero) and finish episode
            if not success and not self.sensor_only:
                joint_goal = self.move_group.get_current_joint_values()
                joint_goal[0] = -0.85
                joint_goal[1] = -2.0
                joint_goal[2] = 2.0
                joint_goal[3] = 0.67
                joint_goal[4] = 0.85
                joint_goal[5] = -2.1
                self.move_group.go(joint_goal, wait=True)
                self._episode_done = True

        rospy.logwarn("END Set Action")

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        :return: the stitched observations
        """
        rospy.logwarn("Start Get Observation ==>")

        # Case 1: Reward is gained for every iteration/step
        if not self.use_cumulated_reward:
            # Get reward and current measurement
            self.area_gain = self.evaluate_measurement()
            self.current_poses = [self.get_current_pose()]

            # Round up to first decimal in case of an discretized action space #TODO parameterize or change?
            if self.is_discretized:
                self.current_poses[0].position.x = np.around(
                    self.current_poses[0].position.x, 2)
                self.current_poses[0].position.y = np.around(
                    self.current_poses[0].position.y, 2)
                self.current_poses[0].position.z = np.around(
                    self.current_poses[0].position.z, 2)
                self.current_poses[0].orientation.x = np.around(
                    self.current_poses[0].orientation.x, 2)
                self.current_poses[0].orientation.y = np.around(
                    self.current_poses[0].orientation.y, 2)
                self.current_poses[0].orientation.z = np.around(
                    self.current_poses[0].orientation.z, 2)
                self.current_poses[0].orientation.w = np.around(
                    self.current_poses[0].orientation.w, 2)

            if self.cumulated_steps == -1:
                self.cumulated_steps += 1
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

            # Compose observation of seven entries representing pose and the area gain
            observations = np.array([
                self.current_poses[0].position.x,
                self.current_poses[0].position.y,
                self.current_poses[0].position.z,
                self.current_poses[0].orientation.x,
                self.current_poses[0].orientation.y,
                self.current_poses[0].orientation.z,
                self.current_poses[0].orientation.w, self.area_gain
            ])
        else:
            observations = []
            if self.current_poses:

                # Compose observation of seven entries of all poses during this episode
                for i in range(self.actions_per_step):
                    if self.is_discretized and self.use_grid:
                        self.current_poses[i].position.x = np.around(
                            self.current_poses[i].position.x, 2)
                        self.current_poses[i].position.y = np.around(
                            self.current_poses[i].position.y, 2)
                        self.current_poses[i].position.z = np.around(
                            self.current_poses[i].position.z, 2)

                    observations.extend([
                        self.current_poses[i].position.x,
                        self.current_poses[i].position.y,
                        self.current_poses[i].position.z,
                        self.current_poses[i].orientation.x,
                        self.current_poses[i].orientation.y,
                        self.current_poses[i].orientation.z,
                        self.current_poses[i].orientation.w
                    ])

                # Append the area gain and convert to numpy array
                observations.extend([self.area_gain])
                observations = np.array(observations)
            else:
                self.area_gain = 0
                observations = np.array([0, 0, 0, 0, 0, 0, 1, self.area_gain])

        self.cumulated_area_gain += self.area_gain
        rospy.logwarn("END Get Observation ==> " + str(observations[-1]))
        return observations

    def _is_done(self, observations):
        if not self._episode_done:
            # We check if the maximum amount of allowed steps is reached
            if self.cumulated_area_gain >= self.desired_coverage:
                self._episode_done = True
            elif self.episode_steps >= self.desired_steps:
                self._episode_done = True

        # Print to console if finished and return
        if self._episode_done:
            rospy.logwarn("Episode is done ==>")
        return self._episode_done

    def _compute_reward(self, observations, done):
        # rospy.logwarn("Start Get Reward")

        self.episode_steps += 1
        self.cumulated_steps += 1
        reward = observations[-1]

        rospy.logwarn("End Get Reward ==> " + str(reward))

        return reward

    # Internal TaskEnv Methods
    def evaluate_measurement(self):
        """Evaluating a measurement and compute
           the area gain, which is used to compose the reward

        Args:
            point_cloud (sensor_msgs/PointCloud2): the simulated measurement

        Returns:
            float: the absolute area gain of the given point cloud in cm^2
        """
        point_cloud = self.get_open3d_point_cloud()

        # Remove Ground/Table from point cloud
        # point_cloud_array = np.asarray(point_cloud.points)
        # point_cloud_array = point_cloud_array[np.where(point_cloud_array[:,2] >= 1.03)]
        # point_cloud.points = open3d.utility.Vector3dVector(point_cloud_array)

        if point_cloud.is_empty():
            normalized_area_gain = 0
            rospy.logerr("point cloud is empty")
        else:
            start_time = time.time()

            # Call the Point Cloud Handler to calculate the area gain
            if self.cumulated_point_cloud == None:
                get_area_gain_req = GetAreaGainRequest()
                get_area_gain_req.standalone_pcd = True
                get_area_gain_req.new_pcd = self.convertOpen3dtoROS(
                    point_cloud)
                get_area_gain_resp = self.get_area_gain.call(get_area_gain_req)
                self.cumulated_point_cloud = self.convertROStoOpen3d(
                    get_area_gain_resp.cumulated_pcd)
                if get_area_gain_resp.area_gain > 0:
                    normalized_area_gain = get_area_gain_resp.area_gain / self.workpiece_area
                else:
                    normalized_area_gain = 0

            else:
                get_area_gain_req = GetAreaGainRequest()
                get_area_gain_req.standalone_pcd = False
                get_area_gain_req.new_pcd = self.convertOpen3dtoROS(
                    point_cloud)
                get_area_gain_req.previous_pcd = self.convertOpen3dtoROS(
                    self.cumulated_point_cloud)
                get_area_gain_resp = self.get_area_gain.call(get_area_gain_req)
                self.cumulated_point_cloud = self.convertROStoOpen3d(
                    get_area_gain_resp.cumulated_pcd)
                if get_area_gain_resp.area_gain > 0:
                    normalized_area_gain = get_area_gain_resp.area_gain / self.workpiece_area
                else:
                    normalized_area_gain = 0

                if self.test_mode:
                    if self.is_discretized and self.episode_num > 1:
                        self.test_area_gain(normalized_area_gain)

            end_time = time.time()
            self.pc_handling_time += end_time - start_time
            self.pc_handling_num += 1
            # rospy.logerr("Handling Time: "+str(self.pc_handling_time/self.pc_handling_num))

        return normalized_area_gain

    #############################
    # View pose sampling
    #############################

    def get_viewpoints(self):
        """Read pre-sampled viewpoints

        Returns:
            list: a list of viewpoints consisting of tuples
            of size seven representing the pose (x,y,z,qx,qy,qz,qw)
        """
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('ipa_kifz_viewplanning')
        viewpoint_filename = os.path.join(
            pkg_path, "config",
            "ipa_kifz_viewposes_" + self.workpiece_name + ".csv")

        df = pd.read_csv(viewpoint_filename)
        viewpoint_list = []
        nb_viewpoints = len(df)
        for i in range(nb_viewpoints):
            viewpoint_pose = tuple(
                (np.around(df.loc[i]['x'], 2), np.around(df.loc[i]['y'], 2),
                 np.around(df.loc[i]['z'], 2), df.loc[i]['qx'],
                 df.loc[i]['qy'], df.loc[i]['qz'], df.loc[i]['qw']))

            viewpoint_list.append(viewpoint_pose)

        return viewpoint_list

    def set_grid_limits(self):
        """Compute start and end boundaries for action space
        This is done by getting the workpiece bounding box
        and some manual chosen threshold in xyz direction.
        """
        wp_mesh = stl.mesh.Mesh.from_file(self.workpiece_path)
        minx = maxx = miny = maxy = minz = maxz = None
        for p in wp_mesh.points:
            # p contains (x, y, z)
            if minx is None:
                minx = p[stl.Dimension.X]
                maxx = p[stl.Dimension.X]
                miny = p[stl.Dimension.Y]
                maxy = p[stl.Dimension.Y]
                minz = p[stl.Dimension.Z]
                maxz = p[stl.Dimension.Z]
            else:
                maxx = max(p[stl.Dimension.X], maxx)
                minx = min(p[stl.Dimension.X], minx)
                maxy = max(p[stl.Dimension.Y], maxy)
                miny = min(p[stl.Dimension.Y], miny)
                maxz = max(p[stl.Dimension.Z], maxz)
                minz = min(p[stl.Dimension.Z], minz)

        # TODO: Add variables to task configuration yaml
        epsilon = -0.05
        z_init = 0.3

        # Set lower and upper boundaries
        x0 = minx - epsilon
        xlim = maxx + epsilon
        y0 = miny - epsilon
        ylim = maxy + epsilon
        z0 = maxz + z_init

        # Add world coordinate system
        trafo_world_wp = np.identity(4)
        trafo_world_wp[:3, 3] = self.workpiece_pose[:3]
        trafo_world_wp[:3, :3] = Rotation.from_euler(
            'xyz', self.workpiece_pose[3:]).as_matrix()

        # Convert limits to homogeneuous trafo
        trafo_wp_0 = np.identity(4)
        trafo_wp_0[:3, 3] = [x0, y0, z0]
        trafo_wp_lim = np.identity(4)
        trafo_wp_lim[:3, 3] = [xlim, ylim, z0]

        # Compute min and max ranges
        trafo_world_0 = np.dot(trafo_world_wp, trafo_wp_0)
        trafo_world_lim = np.dot(trafo_world_wp, trafo_wp_lim)
        self.min_range_x = trafo_world_0[0, 3]
        self.min_range_y = trafo_world_0[1, 3]
        self.min_range_z = trafo_world_0[2, 3]
        self.max_range_x = trafo_world_lim[0, 3]
        self.max_range_y = trafo_world_lim[1, 3]
        self.max_range_z = trafo_world_lim[2, 3]

        return

    def get_grid(self):
        """This method samples viewpoints in a regular grid, which is defined by
        its type (triangle or square) and step sizes in different directions 

        Returns:
            array: A discrete list of view poses representing the grid
        """
        # ...in case of triangle grid
        if self.triangle_grid:
            self.step_size_x = sqrt(.75 * self.step_size_y**2)

            x_steps = ceil(
                (self.max_range_x - self.min_range_x) / self.step_size_x) + 1
            y_steps = ceil(
                (self.max_range_y - self.min_range_y) / self.step_size_y) + 1
            z_steps = ceil(
                (self.max_range_z - self.min_range_z) / self.step_size_z) + 1

            # define view angles for pitch
            pitch = 255.0
            # ...and yaw, depending on number of yaws
            yaw_step = 360 / self.steps_yaw
            yaws = []
            for i in range(self.steps_yaw):
                yaws.append((i + .5) * yaw_step)

            poses = []
            for i in range(x_steps):
                for j in range(y_steps):
                    for k in range(z_steps):
                        for l in range(self.steps_yaw):
                            quat = Rotation.from_euler('xyz',
                                                       [0, pitch, yaws[l]],
                                                       degrees=True).as_quat()
                            if (i % 2) == 0:
                                pose = tuple(
                                    (round(
                                        self.min_range_x +
                                        (i - 0.5) * self.step_size_x, 2),
                                     round(
                                         self.min_range_y +
                                         (j - 0.75) * self.step_size_y, 2),
                                     round(
                                         self.min_range_z +
                                         k * self.step_size_z, 2), quat[0],
                                     quat[1], quat[2], quat[3]))
                                poses.append(pose)
                            else:
                                pose = tuple(
                                    (round(
                                        self.min_range_x +
                                        (i - 0.5) * self.step_size_x, 2),
                                     round(
                                         self.min_range_y +
                                         (j - 0.25) * self.step_size_y, 2),
                                     round(
                                         self.min_range_z +
                                         k * self.step_size_z, 2), quat[0],
                                     quat[1], quat[2], quat[3]))
                                poses.append(pose)

        # in case of a square grid:
        else:
            # New approach with angle change
            x_steps = ceil(
                (self.max_range_x - self.min_range_x) / self.step_size_x) + 1
            y_steps = ceil(
                (self.max_range_y - self.min_range_y) / self.step_size_y) + 1
            z_steps = ceil(
                (self.max_range_z - self.min_range_z) / self.step_size_z) + 1

            # define view angles for pitch
            pitch = 255.0
            # ...and yaw, depending on number of yaws
            yaw_step = 360 / self.steps_yaw
            yaws = []
            for i in range(self.steps_yaw):
                yaws.append((i + .5) * yaw_step)

            poses = []
            for i in range(x_steps):
                for j in range(y_steps):
                    for k in range(z_steps):
                        for l in range(self.steps_yaw):
                            quat = Rotation.from_euler('XYZ',
                                                       [0, pitch, yaws[l]],
                                                       degrees=True).as_quat()
                            pose = tuple(
                                (round(
                                    self.min_range_x +
                                    (i - .5) * self.step_size_x, 2),
                                 round(
                                     self.min_range_y +
                                     (j - .5) * self.step_size_y, 2),
                                 round(self.min_range_z + k * self.step_size_z,
                                       2), quat[0], quat[1], quat[2], quat[3]))
                            poses.append(pose)

        return poses

    #############################
    # Workpiece Handling
    #############################

    def workpiece_handler(self):
        """Add a workpiece from some dataset by its name and initial position
        """
        self.add_workpiece("test_dataset", self.workpiece_name,
                           self.workpiece_pose[0], self.workpiece_pose[1],
                           self.workpiece_pose[2], self.workpiece_pose[3],
                           self.workpiece_pose[4], self.workpiece_pose[5])

    def add_workpiece(self, dataset, workpiece, x, y, z, roll, pitch, yaw):
        """Spawns the workpiece at the given pose.
        If necessary, sample a point cloud from the mesh
        Add the mesh as MoveIt collision environment for path planning as well

        """
        # Add Workpiece to Gazebo using a Launch file
        ROSLauncher(rospackage_name="ipa_kifz_viewplanning",
                    launch_file_name="spawn_workpiece.launch",
                    arguments="dataset:=" + dataset + " " + "object:=" +
                    workpiece + " " + "x:=" + str(x) + " " + "y:=" + str(y) +
                    " " + "z:=" + str(z) + " " + "R:=" + str(roll) + " " +
                    "P:=" + str(pitch) + " " + "Y:=" + str(yaw))

        # Load mesh and transform to world coordinate system
        rospack = rospkg.RosPack()
        dataset_path = rospack.get_path('ipa_kifz_data')
        self.workpiece_path = os.path.join(dataset_path, dataset, "meshes",
                                           workpiece + ".STL")
        workpiece_pcd_path = os.path.join(dataset_path, dataset, "pointclouds",
                                          workpiece + ".pcd")
        self.workpiece_mesh = open3d.io.read_triangle_mesh(self.workpiece_path)
        self.workpiece_area = open3d.geometry.TriangleMesh.get_surface_area(
            self.workpiece_mesh)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        T[:3, 3] = [x, y, z]
        self.workpiece_mesh = self.workpiece_mesh.transform(T)

        # Sample Mesh and save result
        if not os.path.exists(workpiece_pcd_path):
            print("Sampling workpiece mesh")
            self.workpiece_pcd, self.workpiece_voxel_length = self.sample_wp(
                self.workpiece_mesh)
            open3d.io.write_point_cloud(workpiece_pcd_path, self.workpiece_pcd)
        else:
            self.workpiece_pcd = open3d.io.read_point_cloud(workpiece_pcd_path)
            _, self.workpiece_voxel_length = self.sample_wp(
                self.workpiece_mesh)

        if not self.sensor_only:
            # Add Workpiece to MoveIt Planning Scene Interface
            workpiece_pose = PoseStamped()
            workpiece_pose.header.frame_id = "world"
            workpiece_pose.pose.position.x = x
            workpiece_pose.pose.position.y = y
            workpiece_pose.pose.position.z = z
            orientation_quat = tf_conversions.transformations.quaternion_from_euler(
                roll, pitch, yaw)
            workpiece_pose.pose.orientation.x = orientation_quat[0]
            workpiece_pose.pose.orientation.y = orientation_quat[1]
            workpiece_pose.pose.orientation.z = orientation_quat[2]
            workpiece_pose.pose.orientation.w = orientation_quat[3]

            self.scene.add_mesh(workpiece, workpiece_pose, self.workpiece_path)
        return

    def sample_wp(self, workpiece):
        """Sample a workpeice

        Args:
            workpiece (open3d.geometry.TriangleMesh): a open3d Triangle Mesh

        Returns:
            [open3d.goemetry.PointCloud]: the sampled point cloud
            [float]: a fixed voxel grid size
        """
        wp_area = open3d.geometry.TriangleMesh.get_surface_area(workpiece)
        point_number = int(wp_area * 300000)
        voxel_size = sqrt(wp_area / point_number)
        wp_pcd = workpiece.sample_points_uniformly(
            number_of_points=point_number)
        return wp_pcd, voxel_size

    #############################
    # Testing and Debugging
    #############################

    def test_area_gain(self, area_gain):
        """This function should investigate, if the point cloud handler
        function is reliable.
        For example, by computing the mean absolute error of rewards
        calculated for same states (or rather same view poses)

        Args:
            area_gain ([type]): [description]
        """
        rospack = rospkg.RosPack()
        log_path = rospack.get_path('ipa_kifz_viewplanning')
        logfilename = os.path.join(log_path, "test_results",
                                   "test_area_gain.txt")

        self.area_gain_control[self.episode_steps - 1] += area_gain
        if os.path.isfile(logfilename):
            with open(logfilename, 'a') as filehandle:
                if self.episode_steps == len(self.area_gain_control):
                    filehandle.write(
                        '%s %s\n\n' %
                        (area_gain,
                         (area_gain -
                          self.area_gain_control[self.episode_steps - 1] /
                          (self.episode_num - 1))))
                else:
                    filehandle.write(
                        '%s %s\n' %
                        (area_gain,
                         (area_gain -
                          self.area_gain_control[self.episode_steps - 1] /
                          (self.episode_num - 1))))
        else:
            with open(logfilename, 'w') as filehandle:
                filehandle.write(
                    '%s %s\n' %
                    (area_gain,
                     (area_gain -
                      self.area_gain_control[self.episode_steps - 1] /
                      (self.episode_num - 1))))
