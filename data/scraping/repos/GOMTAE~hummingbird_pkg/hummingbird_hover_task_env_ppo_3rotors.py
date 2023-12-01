#! /usr/bin/env python

import rospy
import numpy
from gym import spaces
import hummingbird_hover_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
max_episode_steps = 1000 # Can be any Value

register(
    id='HummingbirdHoverTaskEnvPPO-v2',
    entry_point='hummingbird_hover_task_env_ppo_3rotors:HummingbirdHoverTaskEnv', #change this part as well when changing env
    max_episode_steps=max_episode_steps,
)   ## CHECK NAMING HERE

class HummingbirdHoverTaskEnv(hummingbird_hover_env.HummingbirdHoverEnv):
    def __init__(self):
        """
        Make hummingbird learn how to hover (Get to a point)
        """

        # Launch ROS
        # ros_ws_abspath = rospy.get_param("/hummingbird/ros_ws_abspath", None)
        # assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        # assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
        #                                        " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
        #                                        "/src;cd " + ros_ws_abspath + ";catkin_make"
        #
        # ROSLauncher(rospackage_name="rotors_gazebo",
        #             launch_file_name="start_world.launch",
        #             ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="hummingbird_pkg",
                               rel_path_from_package_to_file="config",
                               yaml_file_name="hummingbird_ppo_params_3rotors.yaml")    ### Need change

        # Only variable needed to be set here
        self.num_envs = rospy.get_param('/hummingbird/num_envs')

        self.n_actions = rospy.get_param('/hummingbird/n_actions')

        self.rpm_max = rospy.get_param("/hummingbird/action/rpm_max")
        self.rpm_min = rospy.get_param("/hummingbird/action/rpm_min")

        a_high = numpy.array([self.rpm_max, #r1
                              self.rpm_max, #r2
                              self.rpm_max, #r3
                              self.rpm_max, #r4
                             ])

        a_low = numpy.array([self.rpm_min, #r1
                             self.rpm_min, #r2
                             self.rpm_min, #r3
                             self.rpm_min, #r4
                            ])

        self.action_space = spaces.Box(a_low, a_high)
        # self.action_space = spaces.Discrete(self.n_actions)
    # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        # Actions and Observations
        self.init_motor_speed = rospy.get_param(
            '/hummingbird/init_motor_speed')
        self.range_motor_speed = rospy.get_param(
            '/hummingbird/range_motor_speed')
        self.max_motor_rpm = self.init_motor_speed + self.range_motor_speed
        self.min_motor_rpm = self.init_motor_speed - self.range_motor_speed
        # self.init_linear_speed_vector = Vector3()
        # self.init_linear_speed_vector.x = rospy.get_param(
        #     '/hummingbird/init_linear_speed_vector/x')
        # self.init_linear_speed_vector.y = rospy.get_param(
        #     '/hummingbird/init_linear_speed_vector/y')
        # self.init_linear_speed_vector.z = rospy.get_param(
        #     '/hummingbird/init_linear_speed_vector/z')
        #
        # self.init_angular_turn_speed = rospy.get_param(
        #     '/hummingbird/init_angular_turn_speed')
        self.running_step = rospy.get_param(
            '/hummingbird/running_step')

        #self.init_point = Float64()
        #self.init_point.data = rospy.get_param("/hummingbird/init_position")

        # Get WorkSpace Dimensions
        self.work_space_x_max = rospy.get_param("/hummingbird/work_space/x_max")
        self.work_space_x_min = rospy.get_param("/hummingbird/work_space/x_min")
        self.work_space_y_max = rospy.get_param("/hummingbird/work_space/y_max")
        self.work_space_y_min = rospy.get_param("/hummingbird/work_space/y_min")
        self.work_space_z_max = rospy.get_param("/hummingbird/work_space/z_max")
        self.work_space_z_min = rospy.get_param("/hummingbird/work_space/z_min")

        # Maximum RPY values
        self.max_roll = rospy.get_param("/hummingbird/max_roll")
        self.max_pitch = rospy.get_param("/hummingbird/max_pitch")
        self.max_yaw = rospy.get_param("/hummingbird/max_yaw")

        # Maximum linear v,a and angular w and a -- just define range

        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/hummingbird/desired_pose/x")
        self.desired_point.y = rospy.get_param("/hummingbird/desired_pose/y")
        self.desired_point.z = rospy.get_param("/hummingbird/desired_pose/z")

        self.desired_point_epsilon = rospy.get_param("/hummingbird/desired_point_epsilon")

        # We place the Maximum and minimum values of the observation

        high = numpy.array([numpy.inf,                  #pos.x
                            numpy.inf,                  #pos.y
                            numpy.inf,                  #pos.z
                            numpy.inf,                  #vel.x
                            numpy.inf,                  #vel.y
                            numpy.inf,                  #vel.z
                            numpy.inf,                  #acc.x
                            numpy.inf,                  #acc.y
                            numpy.inf,                  #acc.z
                            self.max_roll,              #att.x
                            self.max_pitch,             #att.y
                            self.max_yaw,               #att.z
                            numpy.inf,                  #ang_vel.x
                            numpy.inf,                  #ang_vel.y
                            numpy.inf,                  #ang_vel.z
                            numpy.inf,                  #m0_rpm
                            numpy.inf,                  #m1_rpm
                            numpy.inf,                  #m2_rpm
                            numpy.inf                   #m3_rpm
                            ])

        low = numpy.array([-numpy.inf,                  #pos.x
                           -numpy.inf,                  #pos.y
                           -numpy.inf,                  #pos.z
                           -numpy.inf,                  #vel.x
                           -numpy.inf,                  #vel.y
                           -numpy.inf,                  #vel.z
                           -numpy.inf,                  #acc.x
                           -numpy.inf,                  #acc.y
                           -numpy.inf,                  #acc.z
                           -1*self.max_roll,            #att.x
                           -1*self.max_pitch,           #att.y
                           -1*self.max_yaw,             #att.z
                           -numpy.inf,                  #ang_vel.x
                           -numpy.inf,                  #ang_vel.y
                           -numpy.inf,                  #ang_vel.z
                           -numpy.inf,                  #m0_rpm
                           -numpy.inf,                  #m1_rpm
                           -numpy.inf,                  #m2_rpm
                           -numpy.inf                   #m3_rpm
                           ])

        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>" + str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" + str(self.observation_space))

        # Rewards (needs shaping)
        self.Ca = rospy.get_param("/hummingbird/reward/Ca")
        self.Cx = rospy.get_param("/hummingbird/reward/Cx")
        self.Cv = rospy.get_param("/hummingbird/reward/Cv")
        self.Comega = rospy.get_param("/hummingbird/reward/Comega")

        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(HummingbirdHoverTaskEnv, self).__init__() #ros_ws_abspath

    def _set_init_pose(self):
        """
        Sets the Robot in its init motor speed
        and lands the robot. Its preparing it to be reseted in the world.
        """
        #raw_input("INIT SPEED PRESS")
        init_motor_input = [self.init_motor_speed, self.init_motor_speed, self.init_motor_speed, self.init_motor_speed]
        self.move_motor(init_motor_input)
        # rospy.sleep(1.0) # wait for some time
        # We Issue the landing command to be sure it starts landing
        # raw_input("LAND PRESS")
        # self.land()

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        #raw_input("TakeOFF PRESS")
        # We TakeOff before sending any movement commands
        # self.takeoff()
        self.motor_speed = self.init_motor_speed
        # For Info Purposes
        self.cumulated_reward = 0.0
        # We get the initial pose to measure the distance from the desired point.
        gt_pose = self.get_base_pose()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(gt_pose.position)

    def _set_action(self, action):
        """
        This set action will Set the motor speed of the hummingbird
        :param action: The action that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class of Hummingbird

        # if action == 0:  # Speed up
        #     self.motor_speed += self.step_motor_speed
        #     # self.last_action = "FORWARDS"
        # elif action == 1:  # Speed down
        #     self.motor_speed -= self.step_motor_speed
        #     # self.last_action = "BACKWARDS"

        # We tell hummingbird the motor speed to set to execute
        # We mask our action here
        max_motor_input = self.max_motor_rpm
        min_motor_input = self.min_motor_rpm
        # print(action)
        motor_input = self.scale_action(action, min_motor_input, max_motor_input)
        motor_input = numpy.round(motor_input, 4)
        self.step_counter += 1
        if self.step_counter > 250 + self.random_motor_failure * 100: # Finetuning
            motor_input[0] = 0
        self.move_motor(motor_input)
        rospy.sleep(self.running_step)  # wait for some time
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have access to, we need to read the
        droneEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the odom sensing data
        gt_pose = self.get_base_pose()
        gt_linear_vel = self.get_base_linear_vel()
        gt_angular_vel = self.get_base_angular_vel()
        gt_linear_acc = self.get_base_linear_acc()
        gt_motor_vel = self.get_base_motor_vel()

        #gt_angular_acc = self.get_base_angular_acc() -- not using angular accerleration

        # We get the orientation of the Drone in RPY
        roll, pitch, yaw = self.get_orientation_euler(gt_pose.orientation)

        # We simplify a bit the spatial grid to make learning faster (round up 5 digit)
        observations = [round(gt_pose.position.x, 5),
                        round(gt_pose.position.y, 5),
                        round(gt_pose.position.z, 5),
                        round(gt_linear_vel.x, 5),
                        round(gt_linear_vel.y, 5),
                        round(gt_linear_vel.z, 5),
                        round(gt_linear_acc.x, 5),
                        round(gt_linear_acc.y, 5),
                        round(gt_linear_acc.z, 5),
                        round(roll, 5),
                        round(pitch, 5),
                        round(yaw, 5),
                        round(gt_angular_vel.x, 5),
                        round(gt_angular_vel.y, 5),
                        round(gt_angular_vel.z, 5),
                        round(gt_motor_vel[0], 5),
                        round(gt_motor_vel[1], 5),
                        round(gt_motor_vel[2], 5),
                        round(gt_motor_vel[3], 5)
                        # round(gt_angular_acc.x, 1),
                        # round(gt_angular_acc.y, 1),
                        # round(gt_angular_acc.z, 1),
                        ]

        rospy.logdebug("Observations==>"+str(observations))
        rospy.logdebug("END Get Observation ==>")
        # print(observations)
        return observations

    def _is_done(self, observations):
        """
        The done can be done due to three reasons:
        1) It went outside the workspace
        2) It flipped due to a crash or something
        3) Timestep termination: Automatic (10s) -- max_episode_steps = 1000
        """

        episode_done = False

        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]
        current_position.z = observations[2]

        current_orientation = Point()
        current_orientation.x = observations[9]
        current_orientation.y = observations[10]
        current_orientation.z = observations[11]

        current_distance = self.get_distance_from_desired_point(current_position)
        # current_velocity = self.
        # is_inside_workspace_now = self.is_inside_workspace(current_position)
        # sonar_detected_something_too_close_now = self.sonar_detected_something_too_close(
        #     sonar_value)
        drone_flipped = self.drone_has_flipped(current_orientation)
        # has_reached_des_point = self.is_in_desired_position(current_position, self.desired_point_epsilon)
        isout = self.is_out(current_position, current_distance, MAX_DISTANCE=3.5)

        # rospy.logwarn(">>>>>> DONE RESULTS <<<<<")
        # if not is_inside_workspace_now:
        #     rospy.logerr("is_inside_workspace_now=" +
        #                  str(is_inside_workspace_now))
        # else:
        #     rospy.logwarn("is_inside_workspace_now=" +
        #                   str(is_inside_workspace_now))
        #
        # if drone_flipped:
        #     rospy.logerr("drone_flipped="+str(drone_flipped))
        # else:
        #     rospy.logwarn("drone_flipped="+str(drone_flipped))

        if isout:
            rospy.logerr("isout="+str(isout))
        else:
            rospy.logwarn("isout="+str(isout))

        # if has_reached_des_point:
        #     rospy.logerr("has_reached_des_point="+str(has_reached_des_point))
        # else:
        #     rospy.logwarn("has_reached_des_point="+str(has_reached_des_point))

        # We see if we are outside the Learning Space
        episode_done = isout or drone_flipped
                       # not(is_inside_workspace_now)
                       # or has_reached_des_point

        if episode_done:
            rospy.logerr("episode_done====>"+str(episode_done))
        else:
            rospy.logwarn("episode_done====>"+str(episode_done))

        return episode_done

    def _compute_reward(self, observations, done):

        reward = 0

        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]
        current_position.z = observations[2]

        current_velocity = Vector3()
        current_velocity.x = observations[3]
        current_velocity.y = observations[4]
        current_velocity.z = observations[5]

        current_ang_velocity = Vector3()
        current_ang_velocity.x = observations[12]
        current_ang_velocity.y = observations[13]
        current_ang_velocity.z = observations[14]

        current_orientation = Point()
        current_orientation.x = observations[9]
        current_orientation.y = observations[10]
        current_orientation.z = observations[11]

        current_acceleration = Vector3()
        current_acceleration.x = observations[6]
        current_acceleration.y = observations[7]
        current_acceleration.z = observations[8]

        current_motor_vel_0 = abs(observations[15])
        current_motor_vel_1 = abs(observations[16])
        current_motor_vel_2 = abs(observations[17])
        current_motor_vel_3 = abs(observations[18])

        drone_flipped = self.drone_has_flipped(current_orientation)
        current_distance = self.get_distance_from_desired_point(current_position)
        current_abs_velocity = self.abs_current_velocity(current_velocity)
        # current_abs_acceleration = self.abs_current_acceleration(current_acceleration)
        # current_motor_vel = [current_motor_vel_0-457, current_motor_vel_1-457, current_motor_vel_2-457, current_motor_vel_3-457]

        # distance_difference = distance_from_des_point - \
        #                       self.previous_distance_from_des_point

        # MAX_DISTANCE = 0.17
        # MAX_VELOCITY = 0.2
        # MAX_ACCELERATION_VAR = 0.5
        #
        #
        # distance_discounted = 1 - ((current_distance / MAX_DISTANCE) ** 0.4)
        # velocity_discounted = 1 - ((current_abs_velocity / MAX_VELOCITY) ** 0.4)
        # z_discounted = 1 - ((abs(9.8-current_acceleration.z) / MAX_ACCELERATION_VAR) ** 0.4)
        # # z_discounted = (1 - max(abs(9.8-current_acceleration.z), 0.001)) ** (1 - max(current_distance, 0.01))
        # # pitch_discounted = (1 - max(abs(current_orientation.x), 0.0001)) ** (1 - max(current_distance, 0.1))
        # # roll_discounted = (1 - max(abs(current_orientation.y), 0.0001)) ** (1 - max(current_distance, 0.1))
        # yaw_rate_discounted = (1 - max(abs(current_ang_velocity.z), 0.001)) ** (1 - max(current_distance, 0.01))

        distance_reward = current_distance * self.Cx
        velocity_reward = current_abs_velocity * self.Cv
        # acceleration_reward = - current_abs_acceleration * 2 * 1e-4
        ang_vel = [current_ang_velocity.x, current_ang_velocity.y, 0]
        # motor_vel_reward = - numpy.linalg.norm(current_motor_vel) * 1e-6
        ang_velocity_reward = numpy.linalg.norm(ang_vel) * self.Comega
        alive_reward = self.Ca

        # re_w = [distance_reward, velocity_reward, ang_velocity_reward]
        # print(re_w)

        # if not done:
        #     reward += 100 * distance_discounted * z_discounted * yaw_rate_discounted * velocity_discounted
        if not done:
            reward += distance_reward + velocity_reward + ang_velocity_reward + alive_reward
            if current_distance < 0.7:
                reward += 0.5
            # if current_distance < 0.03:
            #     self.good_step_counter += 1
            #     if abs(current_acceleration.z) < 9.85 and abs(current_acceleration.z) > 9.75:
            #         reward += (self.good_step_counter ** 2) * z_discounted * 4 * distance_discounted
                    # print(' ')
                    # print('[ZONE] HOVERING')
                    # print(' ')
                # if abs(current_ang_velocity.z) < 0.1:
                #     reward += (self.good_step_counter ** 1.5) * yaw_rate_discounted * distance_discounted
                #     print(' ')
                #     print('[ZONE] GOOD ZONE')
                #     print(' ')
                # print('Good_step_counter: {}'.format(self.good_step_counter))
                # if self.is_in_low_velocity(current_velocity):
                #     reward += (self.good_step_counter ** 2) * distance_discounted * 2

            # else:
            #     self.good_step_counter = 0
            #     # print(' ')
            #     # print('[ZONE] -------------------->!!! ON THE WAY !!!')
            #     # print(' ')
            #     reward += -10 * current_distance

        else:
            if current_distance < 0.03:
                reward += 10
            elif drone_flipped == True:
                reward += -100

        # print('Linear Z: {}'.format(current_acceleration.z))
        print('Current distance: {} | Current z-acceleration: {} | Reward: {} | Done: {}'.format(current_distance, current_acceleration.z, reward, done))
        # print(' ')

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))

        return reward

        # if not done:
        #
        #     # If in the bound & low vel, we reward it
        #     if self.is_in_desired_boundary(current_position) and self.is_in_low_velocity(current_velocity):
        #         if self.is_in_desired_position(current_position):
        #             reward = self.inside_goal_reward
        #         elif self.is_in_desired_position(current_position):
        #             reward = self.inside_boundary_reward
        #         else:
        #             reward = 0
        #     else:
        #         reward = -self.inside_boundary_reward
        #
        # else:
        #     if self.drone_has_flipped(current_orientation):
        #         reward = -5 * self.inside_goal_reward
        #     else:
        #         reward = 5 * self.inside_goal_reward
        #
        # # self.previous_distance_from_des_point = distance_from_des_point
        #
        # rospy.logdebug("reward=" + str(reward))
        # self.cumulated_reward += reward
        # rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        # self.cumulated_steps += self.not_ending_point_reward
        # rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        #
        # return reward

    # Internal TaskEnv Methods

    def scale_action(self, action, min_vel, max_vel):
        return numpy.interp(action, [-1, 1], [min_vel, max_vel])

    def is_in_desired_boundary(self, current_position, bound=0.2):
        """
        It return True if the current position is in the boundary
        """

        x_current = current_position.x
        y_current = current_position.y
        z_current = current_position.z

        return numpy.sqrt((self.desired_point.x - x_current) ** 2 + (self.desired_point.y - y_current) ** 2 + (self.desired_point.z - z_current) ** 2) < bound

    def abs_current_velocity(self, current_velocity):

        v_x = current_velocity.x
        v_y = current_velocity.y
        v_z = current_velocity.z

        return numpy.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)

    def abs_current_acceleration(self, current_acceleration):

        a_x = current_acceleration.x
        a_y = current_acceleration.y
        a_z = current_acceleration.z - 9.8

        return numpy.sqrt(a_x ** 2 + a_y ** 2 + a_z ** 2)


    def is_in_low_velocity(self, current_velocity, bound=0.1):
        """
        It return True if the current position is in the boundary
        """

        v_x = current_velocity.x
        v_y = current_velocity.y
        v_z = current_velocity.z

        return numpy.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2) < bound

    def is_out(self, current_position, current_distance, MAX_DISTANCE):

        isout = False

        if abs(current_position.x) >= 2 or abs(current_position.y) >= 2 or current_distance >= MAX_DISTANCE:
            isout = True

        return isout

    # def is_in_desired_position(self, current_position, epsilon=0.05):
    #     """
    #     It return True if the current position is similar to the desired poistion
    #     """
    #
    #     is_in_desired_pos = False
    #
    #     x_pos_plus = self.desired_point.x + epsilon
    #     x_pos_minus = self.desired_point.x - epsilon
    #     y_pos_plus = self.desired_point.y + epsilon
    #     y_pos_minus = self.desired_point.y - epsilon
    #     z_pos_plus = self.desired_point.z + epsilon
    #     z_pos_minus = self.desired_point.z - epsilon
    #
    #     x_current = current_position.x
    #     y_current = current_position.y
    #     z_current = current_position.z
    #
    #     x_pos_are_close = (x_current <= x_pos_plus) and (
    #             x_current > x_pos_minus)
    #     y_pos_are_close = (y_current <= y_pos_plus) and (
    #             y_current > y_pos_minus)
    #     z_pos_are_close = (z_current <= z_pos_plus) and (
    #             z_current > z_pos_minus)
    #
    #     is_in_desired_pos = x_pos_are_close and y_pos_are_close and z_pos_are_close
    #
    #     rospy.logwarn("###### IS DESIRED POS ? ######")
    #     rospy.logwarn("current_position"+str(current_position))
    #     rospy.logwarn("x_pos_plus"+str(x_pos_plus) +
    #                   ",x_pos_minus="+str(x_pos_minus))
    #     rospy.logwarn("y_pos_plus"+str(y_pos_plus) +
    #                   ",y_pos_minus="+str(y_pos_minus))
    #     rospy.logwarn("z_pos_plus"+str(z_pos_plus) +
    #                   ",z_pos_minus="+str(z_pos_minus))
    #     rospy.logwarn("x_pos_are_close"+str(x_pos_are_close))
    #     rospy.logwarn("y_pos_are_close"+str(y_pos_are_close))
    #     rospy.logwarn("z_pos_are_close"+str(z_pos_are_close))
    #     rospy.logwarn("is_in_desired_pos"+str(is_in_desired_pos))
    #     rospy.logwarn("############")
    #
    #     return is_in_desired_pos

    # def is_inside_workspace(self, current_position):
    #     """
    #     Check if the Drone is inside the Workspace defined
    #     """
    #     is_inside = False
    #
    #     rospy.logwarn("##### INSIDE WORK SPACE? #######")
    #     rospy.logwarn("XYZ current_position"+str(current_position))
    #     rospy.logwarn("work_space_x_max"+str(self.work_space_x_max) +
    #                   ",work_space_x_min="+str(self.work_space_x_min))
    #     rospy.logwarn("work_space_y_max"+str(self.work_space_y_max) +
    #                   ",work_space_y_min="+str(self.work_space_y_min))
    #     rospy.logwarn("work_space_z_max"+str(self.work_space_z_max) +
    #                   ",work_space_z_min="+str(self.work_space_z_min))
    #     rospy.logwarn("############")
    #
    #     if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
    #         if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
    #             if current_position.z > self.work_space_z_min and current_position.z <= self.work_space_z_max:
    #                 is_inside = True
    #
    #     return is_inside

    def drone_has_flipped(self, current_orientation):
        """
        Based on the orientation RPY given states if the drone has flipped
        """
        has_flipped = True

        self.max_roll = rospy.get_param("/hummingbird/max_roll")
        self.max_pitch = rospy.get_param("/hummingbird/max_pitch")

        rospy.logwarn("#### HAS FLIPPED? ########")
        rospy.logwarn("RPY current_orientation"+str(current_orientation))
        rospy.logwarn("max_roll"+str(self.max_roll) +
                      ",min_roll="+str(-1*self.max_roll))
        rospy.logwarn("max_pitch"+str(self.max_pitch) +
                      ",min_pitch="+str(-1*self.max_pitch))
        rospy.logwarn("############")

        if current_orientation.x > -1*self.max_roll and current_orientation.x <= self.max_roll:
            if current_orientation.y > -1*self.max_pitch and current_orientation.y <= self.max_pitch:
                has_flipped = False

        return has_flipped

    def get_base_pose(self):

        odom = self.get_odom()
        base_pose = odom.pose.pose

        return base_pose

    def get_base_linear_vel(self):

        odom = self.get_odom()
        base_linear_vel = odom.twist.twist.linear

        return base_linear_vel

    def get_base_angular_vel(self):

        odom = self.get_odom()
        base_angular_vel = odom.twist.twist.angular

        return base_angular_vel

    def get_base_rpy(self):

        imu = self.get_imu()
        base_orientation = imu.orientation

        euler_rpy = Vector3()
        euler = euler_from_quaternion([ base_orientation.x,
                                        base_orientation.y,
                                        base_orientation.z,
                                        base_orientation.w]
                                      )
        euler_rpy.x = euler[0]
        euler_rpy.y = euler[1]
        euler_rpy.z = euler[2]

        return euler_rpy

    def get_base_linear_acc(self):

        imu = self.get_imu()
        base_linear_acc = imu.linear_acceleration

        return base_linear_acc

    def get_base_motor_vel(self):

        vel = self.get_actuators()
        motor_vel = vel.angular_velocities

        return motor_vel


    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)

        return distance

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw