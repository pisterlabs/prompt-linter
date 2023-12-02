from turtle import done

import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import turtlebot3_env
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import os
from math import pi, fabs, sqrt,atan2, cos, sin, modf
from openai_ros.respawnGoal import Respawn

# Waffle Pi footprint
footprint = [[-0.205, -0.155], [-0.205, 0.155], [0.077, 0.155], [0.077, -0.155]]
right_bottom_corner = footprint[0] # [x,y]
left_bottom_corner = footprint[1]
left_top_corner = footprint[2]
right_top_corner = footprint[3]
GOAL_DECAY = 1/2

class TurtleBot3NavEnv(turtlebot3_env.TurtleBot3Env):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot3/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="turtlebot3_gazebo",
                    launch_file_name="start_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/turtlebot3_nav/config",
                               yaml_file_name="turtlebot3_nav.yaml")


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot3NavEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        # number_actions = rospy.get_param('/turtlebot3/n_actions')
        number_actions = 5
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)


        number_observations = rospy.get_param('/turtlebot3/n_observations')
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

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot3/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot3/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/turtlebot3/new_ranges')
        self.new_ranges = 24
        self.min_range = rospy.get_param('/turtlebot3/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot3/max_laser_value')
        self.max_laser_value = 3.5
        self.min_laser_value = rospy.get_param('/turtlebot3/min_laser_value')
        self.max_linear_aceleration = rospy.get_param('/turtlebot3/max_linear_aceleration')

        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        # laser_scan = self.get_laser_scan()
        # num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)
        num_laser_readings = self.new_ranges

        # self.l1 + self.l2 +  [self._get_distance2goal(), self._get_heading()]
        # high= numpy.full((num_laser_readings * 2 + 2), 1)
        # low= numpy.full((num_laser_readings * 2 + 2), -1)
        # self.l1 +  [self._get_distance2goal(), self._get_heading()]
        high_scan= numpy.full((num_laser_readings), self.max_laser_value)
        low_scan= numpy.full((num_laser_readings ), self.min_laser_value)
        high_pos = numpy.array([10, pi])
        low_pos = numpy.array([0, -pi])
        high =  numpy.concatenate([high_scan, high_pos], axis = 0)
        low =  numpy.concatenate([low_scan, low_pos], axis = 0)

        # We only use two integers
        self.observation_space = spaces.Box(low, high) # just for sample or check contain

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("/turtlebot3/forwards_reward")
        self.turn_reward = rospy.get_param("/turtlebot3/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlebot3/end_episode_points")
        # self.goal_reaching_points = rospy.get_param("/turtlebot2/goal_reaching_points",500)

        # [Tri Huynh]
        # Goal reward
        self.goal_reaching_points = 200
        self.closeness_threshold = 0.2
        self.goal_x, self.goal_y = 0, 0
        self.initGoal = True
        self.respawn_goal = Respawn()

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
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        # [Tri Huynh]

        self._reached_goal = False
        self.previous_distance2goal = self._get_distance2goal()
        discretized_scan = self.discretize_scan_observation(
                                                    self.get_laser_scan(),
                                                    self.new_ranges
                                                    )
        self.l2 = self.l1 = discretized_scan
        self.current_distance2goal = self.init_distance2goal = self._get_distance2goal()
        self.ang_vel = self.get_odom().twist.twist.angular.z
        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self._get_distance2goal()
        self.global_step = 0

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        # if action == 0: #FORWARD
        #     linear_vel = self.linear_forward_speed
        #     angular_speed = 0.0
        # elif action == 1: #LEFT
        #     linear_vel = self.linear_turn_speed
        #     angular_speed = self.angular_speed
        #     # self.last_action = "TURN_LEFT"
        # elif action == 2: #RIGHT
        #     linear_vel = self.linear_turn_speed
        #     angular_speed = -1*self.angular_speed
        #     # self.last_action = "TURN_RIGHT"
        # self.last_action = action
        
        self.global_step +=1
        linear_vel = .2
        max_ang_vel = pi/2
        self.ang_vel = ((self.action_space.n- 1)/2 - action) * max_ang_vel * 0.5

        # We tell TurtleBot3 the linear and angular speed to set to execute
        self.move_base(linear_vel, self.ang_vel, epsilon=0.05, update_rate=10)

        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot3Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        discretized_scan = self.discretize_scan_observation(
                                                    self.get_laser_scan(),
                                                    self.new_ranges
                                                    )
        # [Tri Huynh]: add previous scan, distace, heading
        self.l2 = self.l1
        self.l1 = discretized_scan
        self.current_distance2goal = self._get_distance2goal()

        if (self.current_distance2goal < 0.2):
            self._reached_goal = True

        # position relative to the goal x, y
        # state = self.l1 + self.l2 +  [self.normalize(self.current_distance2goal, 10), round(self._get_heading()/ pi, 2)]
        state = self.l1 +  [self.current_distance2goal, self._get_heading()]
        # rospy.logwarn("STATE: " + str(state))
        return state

    def _is_done(self, observations):

        # if self._episode_done:
        
        # if self._episode_done and (not self._reached_goal):
        #     rospy.logerr("TurtleBot3 is Too Close to wall==>")
        # elif self._episode_done and self._reached_goal:
        #     rospy.logwarn("Robot reached the goal")

        # Now we check if it has crashed based on the imu
        imu_data = self.get_imu()
        linear_acceleration_magnitude = self.get_vector_magnitude(imu_data.linear_acceleration)
        if linear_acceleration_magnitude > self.max_linear_aceleration:
            # rospy.logerr("TurtleBot3 Crashed==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))
            self._episode_done = True

        return self._episode_done

    def _compute_reward(self, observations, action, done):
        return self.setReward2(observations, done, action)
        #  [Tri Huynh] ############################3
        reward = 0
        r_towardgoal = 2
        w_oscillatory = 0.05
        w_H = 3
        if not done:
            heading = self._get_heading()
            if heading > pi/2 or heading < pi/2:
                rH = - 10 ** (-(pi - fabs(heading)))
            else:
                rH = 10 ** (-fabs(heading))
            reward += w_H * rH
            print("reward heading to goal:" + str(rH))

            rG = r_towardgoal*(self.previous_distance2goal - self.current_distance2goal)
            reward += rG
            print("reward to goal:" + str(rG))

            rO = w_oscillatory *-1* fabs(self.ang_vel) if action in [0,4] \
                else w_oscillatory * 1 * (pi/2-fabs(self.ang_vel))
            reward += rO
            print("reward Osci:" + str(rO))
        #############################3

            self.previous_distance2goal = self.current_distance2goal
            if self._reached_goal:
                # reward += self.goal_reaching_points
                reward += 200
                print("Goal !!! ")
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
                self.goal_distance = self._get_distance2goal()
                self._reached_goal = False


        if done:
            # if not self._reached_goal:
                # reward += -1*self.end_episode_points
                reward += -200
                print("Collision !!!: ")
        
        # Danger of collision cost
        # reward += self.collision_danger_cost
        # rospy.logerr("Total reward: " + str(reward))

        return reward

    def setReward(self, state, done, action):
        yaw_reward = []
        heading = self._get_heading()
        current_distance = self._get_distance2goal()

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * fabs(0.5 - modf(0.25 + 0.5 * angle % (2 * pi) / pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)
        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

        if done:
            print("Collision!!")
            reward = -200
            # self.pub_cmd_vel.publish(Twist())

        if self._reached_goal:
            print("Goal!!")
            reward = 200
            # self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self._reached_goal = False

        return reward

    def setReward2(self, state, done, action):
        yaw_reward = []
        heading = self._get_heading()
        current_distance = self._get_distance2goal()
        w_sharp_turn = 5
        w_time = 1

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * fabs(0.5 - modf(0.25 + 0.5 * angle % (2 * pi) / pi)[0])
            yaw_reward.append(tr)

        distance_rate = (current_distance / self.goal_distance + .2)
        rH = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

        norm_ang_vel = fabs(self.ang_vel)
        rST = -w_sharp_turn * norm_ang_vel if norm_ang_vel >= (pi/4 - 0.1) else 0
        reward = rH + rST
        # print(f"reward H: {rH}" )
        # print(f"reward ST: {rST}" )

        if done:
            print("Collision!!")
            rTime = -w_time * (200 - self.global_step) # since 200 is maximum. [TODO: get max_env_step instead of 200]
            rCollision = -200
            reward = rTime + rCollision
            # print("COLLISION reward" + str(reward))
            # self.pub_cmd_vel.publish(Twist())

        if self._reached_goal:
            print("Goal!!")
            reward = 200
            # self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self._reached_goal = False

        return reward
    # Internal TaskEnv Methods
    # [Tri Huynh] min pooling + normalize obs
    def discretize_scan_observation(self, scan, new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        scan_range = []
        kernel_size = len(scan.ranges)// new_ranges
        for i in range(0, len(scan.ranges), kernel_size):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif numpy.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
            if self.min_range > scan.ranges[i]:
                self._episode_done = True

        # min_pooling_scan = []
        # min_dist = self.max_laser_value

        # for i, dist in enumerate(data.ranges):
        #     if dist == float ('Inf') or numpy.isinf(dist): pass
        #     if numpy.isnan(dist): min_dist = self.min_laser_value
        #     if dist < min_dist:
        #         min_dist = dist
        #     if (i % kernel_size==kernel_size - 1): #0 -> kernel_size - 1
        #         min_pooling_scan.append(min_dist)
        #         if min_dist < self.closeness_threshold:
        #             radius_angle = i / 180.0 * pi
        #             # x go up(head), y go left, radius 0 at positive x, counter-clockwise
        #             x = min_dist * cos(radius_angle)
        #             y = min_dist * sin(radius_angle)
        #             vel_x = self.get_odom().twist.twist.linear.x
        #             if self.obsFront(x, y) and vel_x > 0.1: #front obs
        #                 self._episode_done = True #early stop
        #         min_dist = self.max_laser_value
        #         if (self.min_range > dist > 0):
        #             self._episode_done = True

                # if dist == float ('Inf') or numpy.isinf(dist):
                #     discretized_ranges.append(self.max_laser_value)
                # elif numpy.isnan(dist):
                #     discretized_ranges.append(self.min_laser_value)
                # else:
                #     discretized_ranges.append(dist, 2)
                #     # if k==0 or k == self.new_ranges-1: # in front, [TODO]: Detect exact 
                #     # TODO: Add free space for moving obstacle (given V_o, or predict V_o)
                #     if r < self.closeness_threshold:
                #         radius_angle = i / 180.0 * math.pi
                #         # x go up(head), y go left, radius 0 at positive x, counter-clockwise
                #         x = r * math.cos(radius_angle)
                #         y = r * math.sin(radius_angle)
                #         if self.obsFront(x, y): #front obs
                #             self._episode_done = True #early stop
                #         # else:
                #         #     self.collision_danger_cost += self.prox_penalty2 / dist
                #             # if item > self.closeness_threshold:
                #             #     self.collision_danger_cost += self.prox_penalty1 / item
                #             # else:
                #             #     self.collision_danger_cost += self.prox_penalty2 / item

        # return [self.normalize(x, self.max_laser_value) for x in discretized_ranges]
        # return min_pooling_scan
        return scan_range

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

    def _get_distance2goal(self):
        """ Gets the distance to the goal
        """
        odom = self.get_odom()
        odom_x, odom_y = odom.pose.pose.position.x, odom.pose.pose.position.y
        return sqrt((self.goal_x - odom_x)**2 + (self.goal_y - odom_y)**2) # NOTE: Distnce Alway less than 10m 

    def _get_heading(self):
        # goal_x, goal_y = self._get_goal_location()
        odom = self.get_odom()
        odom_x, odom_y = odom.pose.pose.position.x, odom.pose.pose.position.y
        # robot_to_goal_x = int((goal_x - odom_x) * 2.4999) # assume maximum is 4, scaled up to 9
        # robot_to_goal_y = int((goal_y - odom_y) * 2.4999)
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        goal_angle = atan2(self.goal_y - odom_y, self.goal_x - odom_x)
        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi
        elif heading < -pi:
            heading += 2 * pi
        return heading 


    def obsBack(self, x, y):
        return x <= 0 and x >= -self.closeness_threshold and y <= left_bottom_corner[1] and y >= right_bottom_corner[1]

    def obsFront(self, x, y):
        return x >= 0 and x <= self.closeness_threshold and y <= left_bottom_corner[1] and y >= right_bottom_corner[1]

    def obsRight(self, x, y):
        return x <= right_top_corner[0] and x >= right_bottom_corner[0] and y >= -self.closeness_threshold and y <= 0
        
    def obsLeft(self, x, y):
        return x <= right_top_corner[0] and x >= right_bottom_corner[0]  and y >= 0 and y <= self.closeness_threshold

    def normalize(self, x, max_value):
        return round(2 * (1 - x/max_value) - 1, 2)
