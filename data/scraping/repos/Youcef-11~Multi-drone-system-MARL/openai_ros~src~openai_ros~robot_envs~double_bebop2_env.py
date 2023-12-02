#!/usr/bin/env python
from openai_ros import robot_gazebo_env
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty
import rospy
import numpy as np


class DoubleBebop2Env(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self):
        self.stop_until = 0

        # Topic names L_BEBOP
        self.L_image_name = "/L_bebop2/camera_base/image_raw"
        self.L_odom_name = "/L_bebop2/ground_truth/odometry"
        self.L_pose_name = "/L_bebop2/ground_truth/pose"


        self.L_cmd_vel_name = "/L_bebop2/cmd_vel"
        self.L_takeoff_name = "/L_bebop2/takeoff"
        self.L_land_name = "/L_bebop2/land"
        self.L_reset_name = "/L_bebop2/fake_driver/reset_pose"

        # Topic names R_BEBOP
        self.R_image_name = "/R_bebop2/camera_base/image_raw"
        self.R_odom_name = "/R_bebop2/ground_truth/odometry"
        self.R_pose_name = "/R_bebop2/ground_truth/pose"

        self.R_cmd_vel_name = "/R_bebop2/cmd_vel"
        self.R_takeoff_name = "/R_bebop2/takeoff"
        self.R_land_name = "/R_bebop2/land"
        self.R_reset_name = "/R_bebop2/fake_driver/reset_pose"



        self.controllers_list = []
        self.robot_name_space = ""
        super(DoubleBebop2Env, self).__init__(controllers_list=self.controllers_list,
                                                robot_name_space=self.robot_name_space,
                                                reset_controls=False,
                                                start_init_physics_parameters = False,
                                                reset_world_or_sim = "WORLD")

        # Relance la physique de gazebo
        self.gazebo.unpauseSim()

        # On regarde si tout est pret
        self._check_all_sensor_ready()


        # SUBSCRIBING
        # rospy.Subscriber(self.L_image_name, Image, self._L_img_cb)
        rospy.Subscriber(self.L_odom_name, Odometry, self._L_odom_cb)
        # rospy.Subscriber(self.L_pose_name, Pose, self._L_pose_cb)

        # rospy.Subscriber(self.R_image_name, Image, self._R_img_cb)
        rospy.Subscriber(self.R_odom_name, Odometry, self._R_odom_cb)
        # rospy.Subscriber(self.R_pose_name, Pose, self._R_pose_cb)


        # Publishers
        rospy.logdebug("Finished subscribing")
        self.L_cmd_pub = rospy.Publisher(self.L_cmd_vel_name, Twist, queue_size=0)
        self.L_land_pub = rospy.Publisher(self.L_land_name, Empty, queue_size=0)
        self.L_takeoff_pub = rospy.Publisher(self.L_takeoff_name, Empty, queue_size=1)
        self.L_reset_pub = rospy.Publisher(self.L_reset_name, Empty, queue_size=1)

        self.R_cmd_pub = rospy.Publisher(self.R_cmd_vel_name, Twist, queue_size=1)
        self.R_land_pub = rospy.Publisher(self.R_land_name, Empty, queue_size=1)
        self.R_takeoff_pub = rospy.Publisher(self.R_takeoff_name, Empty, queue_size=1)
        self.R_reset_pub = rospy.Publisher(self.R_reset_name, Empty, queue_size=1)



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
        # self.L_image_raw = self.check_sensor(self.L_image_name, Image)
        self.L_odom = self.check_sensor(self.L_odom_name, Odometry)
        # self.L_pose = self.check_sensor(self.L_pose_name, Pose)
        
        # self.R_image_raw = self.check_sensor(self.R_image_name, Image)
        self.R_odom = self.check_sensor(self.R_odom_name, Odometry)
        # self.R_pose = self.check_sensor(self.R_pose_name, Pose)
        rospy.logdebug("ALL SENSORS READY")
    
    def _check_all_pub_ready(self):
        self.check_publisher(self.L_cmd_pub)
        self.check_publisher(self.L_takeoff_pub)
        self.check_publisher(self.L_land_pub)

        self.check_publisher(self.R_cmd_pub)
        self.check_publisher(self.R_takeoff_pub)
        self.check_publisher(self.R_land_pub)
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
    # def _L_img_cb(self, data):
    #     self.L_image_raw = data

    def _L_odom_cb(self, data):
        self.L_odom = data



    # def _L_pose_cb(self, data):
    #     self.L_pose = data
        
    # def _R_img_cb(self, data):
    #     self.R_image_raw = data

    def _R_odom_cb(self, data):
        self.R_odom = data

    # def _R_pose_cb(self, data):
    #     self.R_pose = data
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
    def takeoff(self, mode = "both"):
        """Envoi un message Empty dans les publishers des drones en fonction du paramètre mode

        Args:
            mode (str, optional): "L" Pour takeoff bebop L, "R" Pour takeoff Bebop R, "both" fait voler les deux. Defaults to "both".
        """
        assert mode in ("L", "R", "both")

        # self.gazebo.unpauseSim()

        if mode == "L" or mode == "both":
            self.check_publisher(self.L_takeoff_pub)
            self.L_takeoff_pub.publish(Empty())

        if mode == "R" or mode == "both":
            self.check_publisher(self.R_takeoff_pub)
            self.R_takeoff_pub.publish(Empty())

        rospy.sleep(1)
        # self.wait_for_height(heigh_value_to_check=1, smaller_than=False, epsilon=0.05, update_rate=10, mode = mode)

    def reset_pub(self):
        # self.gazebo.unpauseSim()
        self.L_reset_pub.publish(Empty())
        self.R_reset_pub.publish(Empty())
        # self.gazebo.pauseSim()

    def land(self, mode = "both"):
        """Envoi un message Empty dans les publishers des drones en fonction du paramètre mode

        Args:
            mode (str, optional): "L" Pour takeoff bebop L, "R" Pour takeoff Bebop R, "both" fait voler les deux. Defaults to "both".
        """
        assert mode in ("L", "R", "both")

        self.gazebo.unpauseSim()

        if mode == "L" or mode == "both":
            self.check_publisher(self.L_land_pub)
            self.L_land_pub.publish(Empty())

        if mode == "R" or mode == "both":
            self.check_publisher(self.R_land_pub)
            self.R_land_pub.publish(Empty())

        # When it takes of value of height is around 1.3
        # self.wait_for_height(heigh_value_to_check=0.8,
        #                      smaller_than=False,
        #                      epsilon=0.05,
        #                      update_rate=10,
        #                      mode = mode)
        self.gazebo.pauseSim()

    def wait_for_height(self, heigh_value_to_check, smaller_than, epsilon, update_rate, mode = "both"):
        """
        Checks if current height is smaller or bigger than a value
        :param: smaller_than: If True, we will wait until value is smaller than the one given
        """

        assert mode in ("L", "R", "both")

        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0

        rospy.logdebug("epsilon>>" + str(epsilon))


        self.gazebo.unpauseSim()
        while not rospy.is_shutdown() and start_wait_time + 4 > rospy.get_rostime().to_sec():
            if mode == "L" or "both":
                L_current_pose = self.L_odom.pose.pose
                L_current_height = L_current_pose.position.z
            if mode =="R" or "both":
                R_current_pose = self.R_odom.pose.pose
                R_current_height = R_current_pose.position.z

            if smaller_than:
                if mode == "L":
                    takeoff_height_achieved = L_current_height <= heigh_value_to_check

                if mode == "R":
                    takeoff_height_achieved = R_current_height <= heigh_value_to_check
                
                if mode == "both":
                    takeoff_height_achieved = (L_current_height <= heigh_value_to_check) and (R_current_height <= heigh_value_to_check)
            else:

                if mode == "L":
                    takeoff_height_achieved = L_current_height >= heigh_value_to_check

                if mode == "R":
                    takeoff_height_achieved = R_current_height >= heigh_value_to_check
                
                if mode == "both":
                    takeoff_height_achieved = (L_current_height >= heigh_value_to_check) and (R_current_height >= heigh_value_to_check)

            if takeoff_height_achieved:
                rospy.logdebug("Reached Height!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rospy.logdebug("Height Not there yet, keep waiting...")
            rate.sleep()



    def publish_cmd(self,name, lin_x,lin_y,lin_z, ang_z  = 0):
        cmd = Twist()
        cmd.linear.x = lin_x
        cmd.linear.y = lin_y
        cmd.linear.z = lin_z
        cmd.angular.z = ang_z

        if name == "R_bebop2" or name == "both":
            self.check_publisher(self.R_cmd_pub)
            self.R_cmd_pub.publish(cmd)
            rospy.logdebug("R_bebop2 cmd_vel published")

        if name == "L_bebop2" or name == "both":
            self.check_publisher(self.L_cmd_pub)
            self.L_cmd_pub.publish(cmd)
            rospy.logdebug("L_bebop2 cmd_vel published")
        
        # peut etre est il nécessaire d'attendre un peu ici
        rospy.sleep(0.03)
        



    def do_hasardous_move(self, min_altitude = 0.5, stop_chance = 0.2):
        """Avec stop_chance %  de chance, le leader s'arrête pendant 3s
        Sinon, il choisis une action au hasard en faisant attention a ne pas passer en dessous de l'altitude 'min_altitude'
        """

        cmd = Twist()
        if rospy.get_rostime().to_sec() < self.stop_until:
            self.L_cmd_pub.publish(cmd)
            rospy.logwarn("3 seconds pause !")
            return

        if np.random.uniform(0,1,1) < stop_chance/100:
            self.stop_until = rospy.get_rostime().to_sec() + 3 
            self.L_cmd_pub.publish(cmd)

        else: 
            action = np.random.uniform(-1,1,3)
            cmd.linear.x = action[0]
            cmd.linear.y = action[1]

            if self.L_odom.pose.pose.position.z < min_altitude and action[2] < 0:
                cmd.linear.z = 1
                rospy.logwarn("Near to ground!")
            else:
                cmd.linear.z = action[2]

            # cmd.angular.z = action[3]

            self.L_cmd_pub.publish(cmd)

        


    def step(self, action):
        """
        Redefinition de la fonction step por ajotuer hazardous move
        """
        rospy.logdebug("START STEP OpenAIROS")

        self.gazebo.unpauseSim()
        self.do_hasardous_move()
        self._set_action(action)
        self.gazebo.pauseSim()
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {}
        reward = self._compute_reward(obs, done)
        # self.cumulated_episode_reward += reward

        rospy.logdebug("END STEP OpenAIROS")

        return obs, reward, done, info

