import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import walrus_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os


class WalrusStairsEnv(walrus_env.WalrusEnv):
    def __init__(self):
        """
        This Task Env is designed for having the Walrus climb and descend stairs.
        It will learn how to climb stairs without tipping over.
        """
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/walrus/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="walrus_gazebo",
                    launch_file_name="load_stairs.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/walrus/config",
                               yaml_file_name="walrus_stairs.yaml")


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(WalrusStairsEnv, self).__init__(ros_ws_abspath)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)


        #number_observations = rospy.get_param('/walrus/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        # Action parameters
        self.linear_forward_speed_max = rospy.get_param('/walrus/linear_forward_speed_max')
        self.linear_forward_speed_min = rospy.get_param('/walrus/linear_forward_speed_min')
        #self.linear_turn_speed = rospy.get_param('/walrus/linear_turn_speed')
        #self.angular_speed = rospy.get_param('/walrus/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/walrus/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/walrus/init_linear_turn_speed')

        # Set up action space. Potential action/commanded velocity is any value between linear_forward_speed_min and _max
        #number_actions = rospy.get_param('/walrus/n_actions')
        #self.action_space = spaces.Discrete(number_actions)
        self.action_space = spaces.Box(numpy.array([self.linear_forward_speed_min]), numpy.array([self.linear_forward_speed_max]))

        # Observation parameters
        self.new_ranges = rospy.get_param('/walrus/new_ranges')
        self.num_scans = rospy.get_param('/walrus/num_scans')
        self.min_range = rospy.get_param('/walrus/min_range')
        self.max_laser_value = rospy.get_param('/walrus/max_laser_value')
        self.min_laser_value = rospy.get_param('/walrus/min_laser_value')
        #self.num_imu_obs = rospy.get_param('/walrus/num_imu_obs')
        self.max_pitch_orient = rospy.get_param('/walrus/max_pitch_orient')
        self.min_pitch_orient = rospy.get_param('/walrus/min_pitch_orient')
        self.max_pitch_rate = rospy.get_param('/walrus/max_pitch_rate')
        self.min_pitch_rate = rospy.get_param('/walrus/min_pitch_rate')
        self.max_x_disp = rospy.get_param('/walrus/max_x_disp')
        self.min_x_disp = rospy.get_param('/walrus/min_x_disp')
        self.max_linear_acceleration = rospy.get_param('/walrus/max_linear_acceleration')
        self.max_angular_velocity = rospy.get_param('/walrus/max_angular_velocity')

        # Set up observation space
        # We create two arrays based on the range values that will be assigned
        # In the discretization method.
        laser_scan_l = self.get_laser_scan_l()
        laser_scan_r = self.get_laser_scan_r()
        #num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)

        # Define high and low values for the scans
        high_scan = numpy.full((self.new_ranges*self.num_scans), self.max_laser_value)
        low_scan = numpy.full((self.new_ranges*self.num_scans), self.min_laser_value)        

        # Now, define high and low values for the imu measurements in a numpy array
        high_imu = numpy.array([self.max_pitch_orient, self.max_pitch_rate])
        low_imu = numpy.array([self.min_pitch_orient, self.min_pitch_rate])

        # Now, define high and low values for the odometry measurement in a numpy array
        high_disp = numpy.array(self.max_x_disp)
        low_disp = numpy.array(self.min_x_disp)        

        # Define high and low values for all observations, and create the observation space to span
        high = numpy.append(high_scan, high_imu)
        high = numpy.append(high, high_disp)
        low = numpy.append(low_scan, low_imu)
        low = numpy.append(low, low_disp)
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Reward/penalty parameters
        self.stay_alive_reward = rospy.get_param("/walrus/stay_alive_reward")
        self.position_reward = rospy.get_param("/walrus/position_reward")
        self.ang_velocity_threshold = rospy.get_param("/walrus/ang_velocity_threshold")
        self.ang_velocity_reward = rospy.get_param("/walrus/ang_velocity_reward")
        self.forward_velocity_reward = rospy.get_param("/walrus/forward_velocity_reward")
        self.cumulated_steps = 0.0


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0

        # Reset Controller
        #self.controllers_object.reset_controllers()

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the walrus
        based on the action given.
        :param action: The action value; i.e. commanded linear velocity.
        """

        rospy.logdebug("Start Set Action ==>"+str(action))
        linear_speed = action[0]
        angular_speed = 0.0
        self.last_action = action[0]       

        # We tell walrus the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)

        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        WalrusEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan_l = self.get_laser_scan_l()
        laser_scan_r = self.get_laser_scan_r()
        imu_data = self.get_imu()
        odom = self.get_odom()

        discretized_observations_l = self.discretize_scan_observation(laser_scan_l, self.new_ranges)
        discretized_observations_r = self.discretize_scan_observation(laser_scan_r, self.new_ranges)
        imu_observations = [imu_data.orientation.y, imu_data.angular_velocity.y]
        odom_observations = [odom.pose.pose.position.x]
        
        obs = [] # initialize empty list
        obs.extend(discretized_observations_l) # add left scan obs to obs
        obs.extend(discretized_observations_r) # add right scan obs to obs
        obs.extend(imu_observations) # add imu obs to obs
        obs.extend(odom_observations) # add odom obs to obs
        # obs.extend(new_list)

        rospy.logdebug("Observations==>"+str(obs))
        rospy.logdebug("END Get Observation ==>")
        return obs


    def _is_done(self, observations):

        if self._episode_done:
            rospy.logerr("Walrus is Too Close to wall==>")
        else:
            rospy.logwarn("Walrus is NOT close to a wall ==>")

        # Check orientation and angular velocity observations for rollover
        if (observations[16]>self.max_pitch_orient)|(observations[16]<self.min_pitch_orient):
            rospy.logerr("Walrus pitch orientation out of bounds==>"+str(observations[16]))
            self._episode_done = True
        else:
            rospy.logdebug("Walrus pitch orientation in bounds==>"+str(observations[16]))
        
        if (observations[17]>self.max_pitch_rate)|(observations[17]<self.min_pitch_rate):
            rospy.logerr("Walrus angular velocity out of bounds==>"+str(observations[17]))
            self._episode_done = True
        else:
            rospy.logdebug("Walrus pitch velocity in bounds==>"+str(observations[17]))

        # Check to see if robot out of bounds
        if (observations[18]>self.max_x_disp)|(observations[18]<self.min_x_disp):
            rospy.logerr("Walrus x-position out of bounds==>"+str(observations[18]))
            self._episode_done = True
        else:
            rospy.logdebug("Walrus x-position in bounds==>"+str(observations[18]))
        
        # Now we check if it has crashed based on the imu
        imu_data = self.get_imu()

        linear_acceleration_magnitude = self.get_vector_magnitude(imu_data.linear_acceleration)
        if linear_acceleration_magnitude > self.max_linear_acceleration:
            rospy.logerr("Walrus Crashed==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_acceleration))
            self._episode_done = True
        else:
            rospy.logerr("Walrus DIDN'T crash ==>"+str(linear_acceleration_magnitude)+"<"+str(self.max_linear_acceleration))


        return self._episode_done

    def _compute_reward(self, observations, done):

        # Reward for staying up / continuing the training episode
        reward = self.stay_alive_reward

        # Penalty for x odometry being far away from origin (off-center)
        rospy.logdebug("Displacement is " + str(observations[18]) + ", reward is " + str(self.position_reward*observations[18]))
        reward += self.position_reward*abs(observations[18])

        # If angular velocity is below threshold, give a reward
        if abs(observations[17]) < self.ang_velocity_threshold:
            rospy.logdebug("Angular velocity " + str(observations[17]) + " is below threshold, giving reward.")
            reward += self.ang_velocity_reward

        # if not done:
        if self.last_action > 0:
            rospy.logdebug("Forward velocity " + str(self.last_action) + ", giving reward " + str(self.forward_velocity_reward*self.last_action))
            reward += self.forward_velocity_reward*self.last_action
        # else:
        #     reward = self.turn_reward
        # else:
        #     reward = -1*self.end_episode_points


        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward


    # Internal TaskEnv Methods

    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges)/new_ranges


        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    #discretized_ranges.append(int(item))
                    discretized_ranges.append(item)

                # Check if collision occurred
                #if (self.min_range > item > 0):
                #    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                #    self._episode_done = True
                #else:
                #    rospy.logdebug("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))


        return discretized_ranges


    def get_vector_magnitude(self, vector):
        """
        It calculated the magnitude of the Vector3 given.
        This is usefull for reading imu accelerations and knowing if there has been
        a crash
        :return:
        """
        contact_force_np = numpy.array((vector.x, vector.y, vector.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude

