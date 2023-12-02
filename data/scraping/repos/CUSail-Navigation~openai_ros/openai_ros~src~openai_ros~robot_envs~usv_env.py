import numpy as np
import rospy
import time
from openai_ros import robot_gazebo_env
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Bool
from openai_ros.openai_ros_common import ROSLauncher


class USVSimEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all USV Sim environments.
    """

    def __init__(self, ros_ws_abspath):
        """
        Initializes a new USVSimEnv environment.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered useful for AI learning.

        Sensor Topic List:
        * /state: Odometry of the sailboat
        * /move_usv/goal: Odometry of the goal

        Actuators Topic List:
        * /joint_setpoint: Publish the positions of the sail and rudder.

        Reset Topic List:
        * /episode_reset: Publish True when the episode is reset

        Args:
        """
        rospy.loginfo("Start USVSimEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # TODO We launch the ROSlaunch that spawns the robot into the world
        # ROSLauncher(
        #     rospackage_name="usv_sim",
        #     launch_file_name=
        #     "/launch/scenarios_launchs/sailboat_scenario1_spawner.launch",
        #     ros_ws_abspath=ros_ws_abspath)

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        self.robot_name_space = "sailboat"

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(USVSimEnv, self).__init__(controllers_list=self.controllers_list,
                                        robot_name_space=self.robot_name_space,
                                        reset_controls=False,
                                        start_init_physics_parameters=False,
                                        reset_world_or_sim="WORLD")

        rospy.loginfo("USVSimEnv unpause1...")
        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()

        self._check_all_systems_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/sailboat/state", Odometry, self._state_callback)
        rospy.Subscriber("/sailboat/move_usv/goal", Odometry,
                         self._goal_callback)

        self.publishers_array = []
        self._joint_state_pub = rospy.Publisher('/sailboat/joint_setpoint',
                                                JointState,
                                                queue_size=1)
        self.publishers_array.append(self._joint_state_pub)

        self._reset_pub = rospy.Publisher('/sailboat/episode_reset',
                                          Bool,
                                          queue_size=10)
        self.publishers_array.append(self._reset_pub)

        self._hit_waypoint_pub = rospy.Publisher('/sailboat/hit_waypoint',
                                                 Bool,
                                                 queue_size=10)
        self.publishers_array.append(self._hit_waypoint_pub)

        self._check_all_publishers_ready()

        self.gazebo.pauseSim()

        rospy.loginfo("Finished USVSimEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        rospy.loginfo("USVSimEnv check_all_systems_ready...")
        self._check_all_sensors_ready()
        rospy.loginfo("END USVSimEnv _check_all_systems_ready...")
        return True

    def _check_all_sensors_ready(self):
        rospy.loginfo("START ALL SENSORS READY")
        self._check_odom_ready()
        rospy.loginfo("ALL SENSORS READY")

    def _check_odom_ready(self):
        self.state = None
        self.goal = None
        rospy.loginfo("Waiting for /state and /move_usv/goal to be READY...")
        while (self.state is None
               or self.goal is None) and not rospy.is_shutdown():
            try:
                self.state = rospy.wait_for_message("/sailboat/state",
                                                    Odometry,
                                                    timeout=1.0)
                self.goal = rospy.wait_for_message("/sailboat/move_usv/goal",
                                                   Odometry,
                                                   timeout=1.0)
                rospy.loginfo("Current /state and /move_usv/goal READY=>")

            except:
                rospy.logerr(
                    "Current /state or /move_usv/goal not ready yet, retrying")
        return self.state, self.goal

    def _state_callback(self, data):
        self.state = data

    def _goal_callback(self, data):
        self.goal = data

    def _check_all_publishers_ready(self):
        """
        Check that all the publishers are working
        """
        rospy.loginfo("START ALL PUBLISHERS READY")
        for publisher_object in self.publishers_array:
            self._check_pub_connection(publisher_object)
        rospy.loginfo("ALL PUBLISHERS READY")

    def _check_pub_connection(self, publisher_object):

        rate = rospy.Rate(10)  # 10hz
        while publisher_object.get_num_connections(
        ) == 0 and not rospy.is_shutdown():
            rospy.loginfo(
                "No subsribers to publisher_object yet so we wait and try again"
            )
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is reset
                pass
        rospy.loginfo("publisher_object Publisher Connected")

        rospy.loginfo("All Publishers READY")

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
    def set_joints(self, sail_position, rudder_position, time_sleep=1.0):
        """
        Sets the positions of the sail and rudder joints.
        """
        i = 0
        msg = JointState()
        msg.header = Header()
        msg.name = ['rudder_joint', 'sail_joint']
        msg.position = [np.deg2rad(rudder_position), np.deg2rad(sail_position)]
        msg.velocity = []
        msg.effort = []

        self._joint_state_pub.publish(msg)
        i += 1
        self.wait_time_for_execute_movement(time_sleep)

    def wait_time_for_execute_movement(self, time_sleep):
        rospy.sleep(time_sleep)

    def send_reset_signal(self):
        msg = Bool()
        msg.data = True
        self._reset_pub.publish(msg)

    def send_hit_waypoint(self):
        msg = Bool()
        msg.data = True
        self._hit_waypoint_pub.publish(msg)

    def get_state(self):
        return self.state

    def get_goal(self):
        return self.goal
