from sys import setdlopenflags
import numpy
import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import turtlebot2_env   # robot env!
from gym.envs.registration import register
from geometry_msgs.msg import Point
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os

from rospy.core import is_initialized

class TurtleBot2WallEnv(turtlebot2_env.TurtleBot2Env):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot2 in some kind of maze.
        It WILL LEARN HOW TO MOVE AROUND THE MAZE WITHOUT CRASHING.
        """

        #This is the path where the simulation files, the Task and Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot2/ros_ws_path", None)
        assert ros_ws_abspath is not NOne, "you forgot to set ros_ws_abspath \
            in your yaml file of your main RL script.Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The simulation ROS workspace path " + ros_ws_abspath +\
            "DOESNOT exist, execute: mkdir -p" + ros_ws_abspath + \
                "/src;cd " +ros_ws_abspath + ";catkin_make"
        
        ROSLauncher(rospackage_name="turtlebot_gazebo", \
            launch_file_name="start_world.launch", \
                ros_ws_abspath=ros_ws_abspath)

        # laod params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros", \
            rel_path_from_package_to_file="src/openai_ros/task_envs/turtlebot2/config",\
                yaml_file_name="turtlebot2_wall.yaml")

        # here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot2WallEnv, self).__init__(ros_ws_abspath)

        # only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot2/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # we set the reward range, which is not compulsory but here we do it.
        self.reward_range - (-numpy.inf, numpy.inf)


        # number_observations = rospy.get_param('turtlebot2/n_observations)
        """
        we set the observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        # actions and observations
        self.linear_forward_speed = rospy.get_param('/turtlebot2/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot2/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot2/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot2/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot2/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/turtlebot2/new_ranges')
        self.min_range = rospy.get_param('/turtlebot2/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot2/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot2/min_laser_value')

        # Get desired point to get
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param('/turtlebot2/desired_pose/x')
        self.desired_point.y = rospy.get_param('/turtlebot2/desired_pose/y')
        self.desired_point.z = rospy.get_param('/turtlebot2/desired_pose/z')

        # we create two arrays based on the binary values that will be assigned
        # in the discretization method
        laser_scan = self.get_laser_scan()
        rospy.logdebug("laser_scan len ==>" +str(len(laser_scan.ranges)))

        num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)
        high = numpy.full((num_laser_readings), self.max_laser_value)
        low = numpy.full((num_laser_readings, self.min_laser_value))

        # we only use two integers
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE ===>" + str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE ===>" + str(self.observation_space))

        # rewards
        self.forwards_reward = rospy.get_param("/turtlebot2/forwards_reward")
        self.turn_reward = rospy.get_param("/turtlebot2/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlebot2/end_episode_points")

        self.cumulated_steps = 0.0

    
    def _set_init_pose(self):
        """sets the robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,\
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
        # For Info purposes
        self.cumulated_reward = 0.0
        # set to false DOne, because its calculated asyncronously
        self._episode_done = False

        odometry = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position)


    def _set_action(self, action):
        """
        thsi set action will set the linear and angular speed of the turtlebot2 
        based on the action number given
        param action: the action integer that set s what movement to do next.
        """

        rospy.logdebug("start set action ==> " + str(action))
        # we convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: # FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action== 1: # LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: # RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action - "TURN_RIGHT"

        
        # we tell turtlebot2 the linear and the angular speed to set to execute
        self.move_base(inear_speed, angular_speed, epsilon=0.05, update_rate=10)

        rospy.logdebug("END set action ==>" +str(action))

    def _get_obs(self):
        """
        here we define what sensor data defined our robots observations 
        to know which variables we have access to, we need to read the 
        TurtleBot2Env API DOCS
        :RETURN:
        """
        rospy.logdebug("START Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()

        discretized_laser_scan = self.discretize_observation(laser_scan, self.new_ranges)

        # we get the odometry so that SumitXL knows where it is 
        odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.posution.y

        # we round to only twp decimals to avoid very big observation space
        odometry_array = [round(x_position,2), round(y_position,2)]

        # we only want the x and y position and the yaw
        observations = discretized_laser_scan + odometry_array
        rospy.logdebug("observations ==>" +str(observations))
        rospy.logdebug("END get observation==>")
        return observations

    def _is_done(self, observations):

        if self._episode_done:
            rospy.logerr("TurtleBot2 is too close to wall==>")
        else:
            rospy.logerr("TurtleBot2 didnt crash at least ==>")
        
            current_position = Point()
            current_position.x = observations[-2]
            current_position.y = observations[-1]
            current_position.z = 0.0

            MAX_X = 6.0
            MIN_X = -1.0
            MAX_Y = 3.0
            MIN_Y = -3.0

            # WE see if we are outside the learning space

            if current_position.x <= MAX_X and current_position.x >MIN_X:
                if current_position.y <= MAX_Y and current_position.y > MIN_Y:
                    rospy.logdebug("turtlebot position is OK ==> [" +str(current_position.x)+","+str(current_position.y)+"]")

                    # we see if it got to the desired point
                    if self.is_in_desired_position(current_position):
                        self._episode_done=True
                
                else:
                    rospy.logerr("turtlebot to Far in Y Pos ==>" +str(current_position.y))
                    self._episode_done = True
            else:
                rospy.logerr("turtlebot to Far in Y Pos ==>" +str(current_position.x))
                self._episode_done = True

        return self._episode_done

    def _compute_reward(self, observations, done):
        current_position = Point()
        current_position.x = observations[-2]
        current_position.y = observations[-1]
        current_position.z = 0.0

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        distance_difference = distance_from_des_point - self.previous_distance_from_des_point

        if not done:

            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward
            
            # id there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                rospy.logwarn("DECREASE IN DISTANCE GOOD")
                reward += self.forwards_reward
            else:
                rospy.logerr("ENCREASE IN DISTANCE BAS")
                reward +=0
        
        else:
            if self.is_in_desired_position(current_position):
                reward = self.end_episode_points
            else:
                reward = -1*self.end_episode_points
        
        self.previous_distance_from_des_point = distance_from_des_point

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward="+str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("cumulated_steps=" +str(self.cumulated_steps))

        return reward

    # internal TaskEnv Methods

    def discretize_observation(Self, data, new_ranges):
        """
        discards all the laser readings that are not multiplw in index of new_ranges value
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges)/new_ranges

        rospy.logdebug("data=" + str(data))
        rospy.logwarn("new_ranges=" + str(new_ranges))
        rospy.logwarn("mod=" + str(mod))  

        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))
                
                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logwarn("NOT done Validation>>>item="+ srt(item)+"< "+str(self.min_range))
        
        return discretized_ranges

    def is_in_desired_position(self, current_position, epsilon=0.05):
        """
        it return True if the current position is similar to the desired position
        """

        is_in_desired_pos = False

        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close
        return is_in_desired_pos

    
    def get_distance_from_desired_point(self, current_position):
        """calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_desired_point(current_position,self.desired_point)
        return distance
    
    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position 
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x. pstart.y, pstart.z))
        b = numpy.array(p_end.x, p_end.y, p_end.z)
        distance = numpy.linalg.norm(a-b)

        return distance
