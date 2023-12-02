import rospy
import numpy
import time
from gym import spaces
from openai_ros.robot_envs import turtlebot2_env
from gym.envs.registration import register
import pdb
import math
import time
from tf.transformations import euler_from_quaternion, quaternion_from_euler

timestep_limit_per_episode = 10000 # Can be any Value

register(
        id='MyTurtleBot2Maze-v0',
        entry_point='my_turtlebot2_maze:MyTurtleBot2MazeEnv',
        timestep_limit=timestep_limit_per_episode,
    )

class MyTurtleBot2MazeEnv(turtlebot2_env.TurtleBot2Env):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot2 in some kind of maze.
        It will learn how to move around the maze without crashing.
        """

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot2/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)


        #number_observations = rospy.get_param('/turtlebot2/n_observations')
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
        self.nsteps = rospy.get_param('/turtlebot2/nsteps')
        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot2/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot2/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot2/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot2/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot2/init_linear_turn_speed')


        self.number_of_sectors = rospy.get_param('/turtlebot2/number_of_sectors')
        self.min_range = rospy.get_param('/turtlebot2/min_range')
        self.middle_range = rospy.get_param('/turtlebot2/middle_range')
        self.middle_range2 = rospy.get_param('/turtlebot2/middle_range2')
        #self.middle_range3 = rospy.get_param('/turtlebot2/middle_range3')

        self.new_ranges = rospy.get_param('/turtlebot2/new_ranges')
        self.danger_laser_value = rospy.get_param('/turtlebot2/danger_laser_value')
        self.middle_laser_value = rospy.get_param('/turtlebot2/middle_laser_value')
        self.middle_laser_value2 = rospy.get_param('/turtlebot2/middle_laser_value2')
        #self.middle_laser_value3 = rospy.get_param('/turtlebot2/middle_laser_value3')
        self.safe_laser_value = rospy.get_param('/turtlebot2/safe_laser_value')
        self.previous_y = -1
        self.max_steps_reached = False
        self.goal_found = False
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        high = numpy.full((self.number_of_sectors), self.danger_laser_value)
        low = numpy.full((self.number_of_sectors), self.safe_laser_value)

        # We only use two integers
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("/turtlebot2/forwards_reward")
        self.turn_reward = rospy.get_param("/turtlebot2/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlebot2/end_episode_points")

        self.cumulated_steps = 0.0
        self.current_nsteps = 0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(MyTurtleBot2MazeEnv, self).__init__()

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
        self.previous_y = -1
        self.max_steps_reached = False
        self.current_nsteps = 0
        self.goal_found = False
        # This is necessary to give the laser sensors to refresh in the new reseted position.
        rospy.logwarn("Waiting...")
        time.sleep(0.5)
        rospy.logwarn("END Waiting...")


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            #linear_speed = 0
            angular_speed = self.angular_speed
            #angular_speed = radians(90)
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            #linear_speed = 0
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            #angular_speed =
            self.last_action = "TURN_RIGHT"
        elif action == 3: #BACKWARDS
            linear_speed = -self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "BACKWARDS"

        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.0, update_rate=100)
        #time.sleep(1)
        #self.move_base(0.0, 0.0, epsilon=0.0, update_rate=100)
        rospy.logdebug("END Set Action ==>"+str(action))
    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()

        discretized_observations = self.discretize_observation( laser_scan,
                                                                self.new_ranges
                                                                )
        odometry = self.get_odom()
        (roll, pitch, yaw) = euler_from_quaternion([odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y, odometry.pose.pose.orientation.z, odometry.pose.pose.orientation.w])
        rospy.logwarn("yaw: "+str(yaw))
        if yaw < -(math.pi*3)/4 or yaw > (math.pi*3)/4:
            ori = 'S'
        elif yaw >= -(math.pi*3)/4 and yaw < -(math.pi)/4:
            ori = 'W'
        elif yaw >= -(math.pi)/4 and yaw < math.pi/4:
            ori = 'N'
        else:
            ori = 'E'

        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y
        #ori = abs(odometry.pose.pose.orientation.z)
        x_position += 3
        x_position = int(x_position)
        x_position -= 2
        discretized_observations.append(x_position)
        discretized_observations.append(ori)
        #discretized_observations.append(int(round(ori,1)))
        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")

        return discretized_observations


    def _is_done(self, observations):
        # 3 things can end an episode:
        # -crashing
        # -reaching max number of steps
        # -reaching the goal
        # here we check for all three

        # did we crash
        episode_done = not (self.check_laser_sector_readings_safe(observations))

        if episode_done:
            rospy.logerr("TurtleBot2 is Too Close to wall==>")
        else:
            rospy.logerr("TurtleBot2 is Ok ==>")
        rospy.logwarn("csteps/nsteps: " + str(self.current_nsteps) + "/" + str(self.nsteps))
        # did we reach max nr of steps
        if self.current_nsteps > self.nsteps-2:
            episode_done = True
            self.max_steps_reached = True
            rospy.logerr("Max steps reached")

        #did we find goal
        position_y = observations[-2]
        if position_y >= 2:
            episode_done = True
            self.goal_found = True
        return episode_done

    def _compute_reward(self, observations, done):
        position_y = observations[-2]
        orientation = observations[-1]
        if not done:
            if orientation == 'N':
                # facing in correct direction
                if self.last_action == "FORWARDS":
                    reward = self.forwards_reward
                elif self.last_action == "BACKWARDS":
                    reward = -1*self.forwards_reward
                else:
                    #turning
                    reward = self.turn_reward
            elif orientation == 'S':
                # facing in wrong direction
                if self.last_action == "FORWARDS":
                    reward = -1*self.forwards_reward
                elif self.last_action == "BACKWARDS":
                    reward = -1*self.turn_reward
                else:
                    reward = 0
            else:
                # facing in east or west
                if self.last_action == "FORWARDS":
                    reward = 0
                elif self.last_action == "BACKWARDS":
                    reward = 0
                else:
                    reward = 0
        else:
            if self.goal_found:
                reward = self.end_episode_points
            elif self.max_steps_reached:
                reward = -150
            else:
                reward = -1*self.end_episode_points

        diff = position_y - self.previous_y
        if diff == 1:
            reward += 20
        elif diff == -1:
            reward -= 20

        self.previous_y =position_y
        #rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        self.current_nsteps += 1

        return reward


    # Internal TaskEnv Methods

    def discretize_observation(self,laser_data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of number_of_sectors
        value.
        """

        base = len(laser_data.ranges)/new_ranges
        current_sector = -1

        sector_readings = [self.safe_laser_value]*new_ranges
        #pdb.set_trace()
        for i, item in enumerate(laser_data.ranges):
            #rospy.logwarn("#### S ###")
            #rospy.logwarn(str(i))
            #rospy.logwarn(str(item))
            rest_is_zero = (i%base==0)

            if rest_is_zero:
                rospy.logwarn("CHANGE SECTOR="+str(rest_is_zero))
                current_sector += 1
            else:
                rospy.loginfo("NO CHANGE SECTOR="+str(rest_is_zero))


            if numpy.isnan(item):
                rospy.logerr(">>>>>>>>>>>>NAN VALUE=>>>"+str(item))

            elif (self.min_range >= item ):
                sector_readings[current_sector] = self.danger_laser_value
            elif (self.middle_range >= item > self.min_range):
                sector_readings[current_sector] = self.middle_laser_value
            elif (self.middle_range2 >= item > self.middle_range):
                sector_readings[current_sector] = self.middle_laser_value2
            #elif (self.middle_range3 >= item > self.middle_range2):
            #    sector_readings[current_sector] = self.middle_laser_value3


        #pdb.set_trace()
        return sector_readings


    def check_laser_sector_readings_safe(self, laser_sector_readings):
        """
        Checks if all the sector readings have the self.safe_laser_value
        of self.middle_laser_value
        """

        readings_safe = all((c != self.danger_laser_value) for c in laser_sector_readings)
        rospy.logwarn("laser_sector_readings=>>>"+str(laser_sector_readings))
        rospy.logwarn("readings_safe=>>>"+str(readings_safe))

        return readings_safe
