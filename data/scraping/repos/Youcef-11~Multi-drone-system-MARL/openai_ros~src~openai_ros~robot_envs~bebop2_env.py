#!/usr/bin/env python
from openai_ros import robot_gazebo_env
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty
import rospy


class Bebop2env(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self):

        # Topic names 
        self.image_name = "/bebop2/camera_base/image_raw"
        self.odom_name = "/bebop2/ground_truth/odometry"
        self.pose_name = "/bebop2/ground_truth/pose"

        self.cmd_vel_name = "/bebop2/fake_driver/cmd_vel"
        self.takeoff_name = "/bebop2/fake_driver/takeoff"
        self.land_name = "/bebop2/fake_driver/land"


        self.controllers_list = []
        self.robot_name_space = "bebop2"
        super(Bebop2env, self).__init__(controllers_list=self.controllers_list,
                                                robot_name_space=self.robot_name_space,
                                                reset_controls=False,
                                                start_init_physics_parameters = False,
                                                reset_world_or_sim = "WORLD")

        # Relance la physique de gazebo
        self.gazebo.unpauseSim()

        # On regarde si tout est pret
        self._check_all_sensor_ready()


        # Subscribing au topics
        rospy.Subscriber(self.image_name, Image, self._img_cb)
        rospy.Subscriber(self.odom_name, Odometry, self._odom_cb)
        rospy.Subscriber(self.pose_name, Pose, self._pose_cb)


        # Publishers
        rospy.logdebug("Finished subscribing")
        self.cmd_pub = rospy.Publisher(self.cmd_vel_name, Twist, queue_size=1)
        self.land_pub = rospy.Publisher(self.land_name, Empty, queue_size=1)
        self.takeoff_pub = rospy.Publisher(self.takeoff_name, Empty, queue_size=1)

        self._check_all_pub_ready()

        rospy.logdebug("checked_allpub")
        # On met en pause la simulation, c'est maintenant a la task de prendre le relai
        self.gazebo.pauseSim()

        rospy.logdebug("Finished Bebop2 INIT...")


    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    
    

    #### CHECKING IF ALL IS GOOD
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensor_ready()
        self._check_all_pub_ready()
        return True

    def _check_all_sensor_ready(self):
        self.image_raw = self.check_sensor(self.image_name, Image)
        self.odom = self.check_sensor(self.odom_name, Odometry)
        self.pose = self.check_sensor(self.pose_name, Pose)
        rospy.logdebug("ALL SENSORS READY")
    
    def _check_all_pub_ready(self):
        self.check_publisher(self.cmd_pub)
        self.check_publisher(self.takeoff_pub)
        self.check_publisher(self.land_pub)
        rospy.logdebug("ALL PUBLISHERS READY")


    def check_sensor(self, topic_name, topic_type):
        msg = None
        rospy.logdebug(f"Waiting for {topic_name} to be Ready")
        while msg is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message(topic_name, topic_type, timeout=5.0)
                rospy.logdebug(f"Current {topic_name} READY=>")
            except:
                rospy.logerr(f"Current {topic_name} not ready yet, retrying for getting the topic")
        return msg
    
    def check_publisher(self, pub : rospy.Publisher):
        rate = rospy.Rate(10)  
        while pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug(f"No susbribers to {pub.name} so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug(f"{pub.name} Publisher Connected")

        rospy.logdebug(f"{pub.name} Ready")
    


    #### CALLBACKS 
    def _img_cb(self, data):
        self.image_raw = data
    
    def _odom_cb(self,data):
        self.odom = data
    
    def _pose_cb(self, data):
        self.pose = data



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
    def takeoff(self):
        """
        Sends the takeoff command and checks it has taken of
        It unpauses the simulation and pauses again
        to allow it to be a self contained action
        """
        self.gazebo.unpauseSim()
        self.check_publisher(self.takeoff_pub)
        self.takeoff_pub.publish(Empty())

        # When it takes of value of height is around 1.3
        self.wait_for_height(heigh_value_to_check=0.8,
                             smaller_than=False,
                             epsilon=0.05,
                             update_rate=10)
        self.gazebo.pauseSim()


    def wait_for_height(self, heigh_value_to_check, smaller_than, epsilon, update_rate):
        """
        Checks if current height is smaller or bigger than a value
        :param: smaller_than: If True, we will wait until value is smaller than the one given
        """

        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0

        rospy.logdebug("epsilon>>" + str(epsilon))

        while not rospy.is_shutdown():
            current_pose = self.check_sensor(self.pose_name, Pose)

            current_height = current_pose.position.z

            if smaller_than:
                takeoff_height_achieved = current_height <= heigh_value_to_check
                rospy.logdebug("SMALLER THAN HEIGHT...current_height=" +
                              str(current_height)+"<="+str(heigh_value_to_check))
            else:
                takeoff_height_achieved = current_height >= heigh_value_to_check
                rospy.logdebug("BIGGER THAN HEIGHT...current_height=" +
                              str(current_height)+">="+str(heigh_value_to_check))

            if takeoff_height_achieved:
                rospy.logdebug("Reached Height!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rospy.logdebug("Height Not there yet, keep waiting...")
            rate.sleep()


