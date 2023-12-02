import os
import sys
import time

import cv2
import rospy

from cv_bridge import CvBridge
from openai_ros import robot_gazebo_env
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class WamvEnv(robot_gazebo_env.RobotGazeboEnv):
    FRONT_CAMERA = '/wamv/sensors/cameras/middle_right_camera/image_raw'
    RIGHT_CAMERA = '/wamv/sensors/cameras/front_right_camera/image_raw'
    LEFT_CAMERA = '/wamv/sensors/cameras/front_left_camera/image_raw'
    ODOM = '/wamv/odom'

    FRONT_CAMERA_KEY = 'image_middle_right'
    RIGHT_CAMERA_KEY = 'image_front_right'
    LEFT_CAMERA_KEY = 'image_front_left'
    ODOM_KEY = 'odom'

    def __init__(self):
        self.systems = {
            self.FRONT_CAMERA_KEY: [None, self.FRONT_CAMERA, Image],
            self.RIGHT_CAMERA_KEY: [None, self.RIGHT_CAMERA, Image],
            self.LEFT_CAMERA_KEY: [None, self.LEFT_CAMERA, Image],
            self.ODOM_KEY: [None, self.ODOM, Odometry]
        }
        self.bridge = CvBridge()

        self.image_callback_iter = 0

        rospy.logdebug(f'Starting {type(self).__name__} INIT')
        # We launch the ROSlaunch that spawns the robot into the world
        # ROSLauncher(rospackage_name="vrx_gazebo",
        #             launch_file_name="put_wamv_in_world.launch",
        #             ros_ws_abspath=ros_ws_abspath)

        # from vrx_gazebo.msg import UsvDrive

        self.controllers_list = []
        self.robot_name_space = ''

        super().__init__(controllers_list=self.controllers_list,
                         robot_name_space=self.robot_name_space,
                         reset_controls=False,
                         start_init_physics_parameters=False,
                         reset_world_or_sim="WORLD")

        rospy.logdebug(f'{type(self).__name__} unpause1')
        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        self._check_all_systems_ready()

        rospy.Subscriber(self.ODOM, Odometry, self._odom_callback)
        rospy.Subscriber(self.FRONT_CAMERA, Image, self._image_middle_right_callback)
        rospy.Subscriber(self.RIGHT_CAMERA, Image, self._image_front_right_callback)
        rospy.Subscriber(self.LEFT_CAMERA, Image, self._image_front_left_callback)

        self.publishers_array = []
        self._cmd_drive_left = rospy.Publisher('/wamv/thrusters/left_thrust_cmd',
                                               Float32,
                                               queue_size=1)
        self._cmd_drive_right = rospy.Publisher('/wamv/thrusters/right_thrust_cmd',
                                                Float32,
                                                queue_size=1)

        self.publishers_array.append(self._cmd_drive_left)
        self.publishers_array.append(self._cmd_drive_right)

        self._check_all_publishers_ready()

        self.gazebo.pauseSim()

        rospy.logdebug(f'Finished {type(self).__name__} INIT')

    def odom(self):
        return self.systems[self.ODOM_KEY][0]


    def image_front(self):
        return self.systems[self.FRONT_CAMERA_KEY][0]


    def image_left(self):
        return self.systems[self.LEFT_CAMERA_KEY][0]


    def image_right(self):
        return self.systems[self.RIGHT_CAMERA_KEY][0]


    def _check_all_systems_ready(self):
        rospy.logdebug(f'{type(self).__name__} check_all_systems_ready')
        self._check_all_sensors_ready()
        rospy.logdebug(f'END {type(self).__name__} _check_all_systems_ready')
        return True


    def _check_all_sensors_ready(self):
        rospy.logdebug("START check_all_sensors_ready")
        for system in self.systems:
            self._check_system_ready(system)
        rospy.logdebug("all sensors ready")
        return True


    # def _check_odom_ready(self):
    #     self.odom = None
    #     rospy.logdebug(f'Waiting for {self.ODOM} to be READY')
    #     while self.odom is None and not rospy.is_shutdown():
    #         try:
    #             self.odom = rospy.wait_for_message(self.ODOM, Odometry, timeout=1.0)
    #             rospy.logdebug(f'Current {self.ODOM} READY')
    #         except:
    #             rospy.logerr(f'Current {self.ODOM} not ready yet, retrying...')
    #     return self.odom


    # def _check_front_camera_ready(self):
    #     rospy.logdebug(f'Waiting for {self.FRONT_CAMERA} to be READY')
    #     while self.image_middle_right is None and not rospy.is_shutdown():
    #         try:
    #             self.image_middle_right =\
    #                 rospy.wait_for_message(self.FRONT_CAMERA, Image, timeout=5.0)
    #             rospy.logdebug('Front camera READY')
    #         except:
    #             rospy.logerr('Front camera NOT READY, retrying...')
    #     return self.image_middle_right


    def _check_system_ready(self, system, timeout=5.0):
        topic = self.systems[system][1]
        rospy.logdebug(f'Waiting for {topic} to be READY')
        while self.systems[system][0] is None and not rospy.is_shutdown():
            try:
                self.systems[system][0] = rospy.wait_for_message(
                    topic,
                    self.systems[system][2],
                    timeout
                )
                rospy.logdebug(f'{topic} READY')
            except:
                rospy.logerr(f'{topic} NOT READY, retrying...')
                time.sleep(0.5)


    def _odom_callback(self, data):
        self.systems['odom'][0] = data


    def _image_callback(self, data, camera):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        self.systems[camera][0] = cv_image
        self.image_callback_iter += 1


    def _image_middle_right_callback(self, data):
        self._image_callback(data, self.FRONT_CAMERA_KEY)


    def _image_front_right_callback(self, data):
        self._image_callback(data, self.RIGHT_CAMERA_KEY)


    def _image_front_left_callback(self, data):
        self._image_callback(data, self.LEFT_CAMERA_KEY)


    def _check_all_publishers_ready(self):
        rospy.logdebug('START check_all_publishers_ready')
        for publisher_object in self.publishers_array:
            self._check_pub_connection(publisher_object)
        rospy.logdebug('END check_all_publishers_ready')


    def _check_pub_connection(self, publisher_object):
        rate = rospy.Rate(10)  # 10hz
        while publisher_object.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to publisher_object yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("publisher_object Publisher Connected")

        rospy.logdebug("All Publishers READY")


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

    def set_propellers_speed(self, right_propeller_speed, left_propeller_speed, time_sleep=1.0):
        rospy.logdebug(f'usv_drive_left >> {left_propeller_speed}')
        rospy.logdebug(f'usv_drive_right >> {right_propeller_speed}')
        self.publishers_array[0].publish(left_propeller_speed)
        self.publishers_array[1].publish(right_propeller_speed)
        self.wait_time_for_execute_movement(time_sleep)

    def wait_time_for_execute_movement(self, time_sleep):
        time.sleep(time_sleep)
