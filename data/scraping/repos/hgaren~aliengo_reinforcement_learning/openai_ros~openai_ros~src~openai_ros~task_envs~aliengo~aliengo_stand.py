'''
Openai-ROS Aliengo Task Environment
Aliengo's actions,states, rewards are evaluated in this Task Environment.
In this case Aliengo needs to stand up and achive desired goal pose.
Garen Haddeler
12.11.2020

'''

import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import aliengo_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os
import math
from laikago_msgs.msg import LowState
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
class AliengoStandEnv(aliengo_env.AliengoEnv):
    def __init__(self):
   

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/aliengo/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"
     
        LoadYamlFileParamsTest(rospackage_name="learning_ros",
                       rel_path_from_package_to_file="config",
                       yaml_file_name="aliengo_stand.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(AliengoStandEnv, self).__init__(ros_ws_abspath)
        
        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/aliengo/desired_pose/x")
        self.desired_point.y = rospy.get_param("/aliengo/desired_pose/y")
        self.desired_point.z = rospy.get_param("/aliengo/desired_pose/z")
        self.desired_pitch = rospy.get_param("/aliengo/desired_pose/pitch")
        self.time_out = rospy.get_param("/aliengo/time_out")
        #Paramater initializations
        self._episode_done = False
        self.joint_increment_value = 0.1
        self.end_episode_points = 100
        self.starting_time =rospy.Time.now().to_sec()
        self.previous_pitch = 0
        self.previous_height = 0
        self.pub = rospy.Publisher("/goal_marker", MarkerArray, queue_size = 1)
        

    def _set_init_pose(self):
        """
        Here we sets the Robot's init pose
        """
        self.paramInit()
        self.starting_time =rospy.Time.now()

        self.pos = [0.0, 0.67, -1.3, -0.0, 0.67, -1.3,0.0, 0.67, -1.3, -0.0, 0.67, -1.3]
        self.moveAllPosition(self.pos)
        #Publish yellow arrow as goal
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "/odom"
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.id = 1
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.position.x = self.desired_point.x
        marker.pose.position.y = self.desired_point.y
        marker.pose.position.z = self.desired_point.z 
        q=self.euler_to_quaternion(0,self.desired_pitch,0)
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
        markerArray.markers.append(marker)
        self.pub.publish(markerArray)

        return True
    def _init_env_variables(self):
        """
        Here we initilize variables
        """
        # For Info Purposes
        self.cumulated_reward = 0.0

        # We get the initial pose to mesure the distance from the desired point.
        odom = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odom.pose.pose.position,self.desired_point)

    def _set_action(self, action):
        """
        Here we convert the actions to joint movements
        """
        if action == 0: #Increment FR FL thigh
            self.pos[1] = self.pos[1] + self.joint_increment_value
            self.pos[4] = self.pos[4] + self.joint_increment_value

        elif action == 1: #Decrease FR FL thigh
            self.pos[1] = self.pos[1] - self.joint_increment_value
            self.pos[4] = self.pos[4] - self.joint_increment_value

        elif action == 2: #Increment FR FL calf
            self.pos[2] = self.pos[2] + self.joint_increment_value
            self.pos[5] = self.pos[5] + self.joint_increment_value

        elif action == 3: #Decrease FR FL calf
            self.pos[2] = self.pos[2] - self.joint_increment_value
            self.pos[5] = self.pos[5] - self.joint_increment_value

        elif action == 4: #Increment RR RL thigh
            self.pos[7] = self.pos[7] + self.joint_increment_value  
            self.pos[10] = self.pos[10] + self.joint_increment_value  

        elif action == 5: #Decrease RR RL thigh
            self.pos[7] = self.pos[7] - self.joint_increment_value  
            self.pos[10] = self.pos[10] - self.joint_increment_value  
        
        elif action == 6: #Increment RR RR calf
            self.pos[8] = self.pos[8] + self.joint_increment_value  
            self.pos[11] = self.pos[11] + self.joint_increment_value  
        
        elif action == 7: #Decrease RR RR calf
            self.pos[8] = self.pos[8] - self.joint_increment_value  
            self.pos[11] = self.pos[11] - self.joint_increment_value  
        #send desired pose to controller
        self.moveAllPosition(self.pos)

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        x: observations[0]  z: observations[1]  pitch: observations[2]
        """
        # We get the odometry
        odometry = self.get_odom()
        (roll, pitch, yaw) = self.quaternion_to_euler_angle (w=odometry.pose.pose.orientation.w,x=odometry.pose.pose.orientation.x,y=odometry.pose.pose.orientation.y,z=odometry.pose.pose.orientation.z)
        # We round to only two decimals to avoid very big Observation space
        observations = [round(odometry.pose.pose.position.x, 2),round(odometry.pose.pose.position.z, 2),round(pitch, 2)]
        return observations

    def _is_done(self, observations):
        """
        Here we check if robot is crashed, get out from bounded zone, time out or achived goal pose
        """
        odometry = self.get_odom()
        self._episode_done = False

        if(self.has_crashed(odometry)):
            self._episode_done = True
            rospy.logerr("Aliengo Crushed==>")
        else:

            current_position = Point()
            current_position.x = observations[0]
            current_position.y = 0
            current_position.z = observations[1]

            MAX_X = 0.7
            MIN_X = -0.8
            MAX_Z = 2.0
            MIN_Z = 0.2

            # We see if we are outside the Learning Space

            if current_position.x <= MAX_X and current_position.x > MIN_X:
                if current_position.z <= MAX_Z and current_position.z > MIN_Z:
                    #print("Position is OK ==>["+str(observations[0])+","+str(observations[1])+","+str(observations[2])+"]")
                    # We see if it got to the desired point
                    if self.is_in_desired_position(observations):
                        self._episode_done = True
                        print("ROBOT REACHED !")  
                else:
                    rospy.logerr("Far in Z Pos ==>"+str(current_position.z))
                    self._episode_done = True
            else:
                rospy.logerr("Far in X Pos ==>"+str(current_position.x))
                self._episode_done = True
                
        end_time =(rospy.Time.now()-self.starting_time).to_sec()

        if(end_time > self.time_out):
            rospy.logerr( "Time Out!")
            print (str(end_time) + " ")
            self._episode_done = True

        return self._episode_done

    def _compute_reward(self, observations, done):
        """
        Here we compute rewards according to our observations
        """
        current_position = Point()
        #print("observation x:"+str(observations[0])+ " z: "+str(observations[1])+ " pitch: "+str(observations[2]))
        current_position.x = observations[0]
        current_position.y =0
        current_position.z = observations[1]

        #Ecludian distance is found from current postion to desired postion
        distance_from_des_point = self.get_distance_from_desired_point(current_position, self.desired_point)
        #delta z and delta pitch angle is found
        delta_height = abs(current_position.z  - self.desired_point.z)
        delta_pitch =  abs(observations[2]-self.desired_pitch)

        #previous difference is calculated
        distance_difference =  distance_from_des_point - self.previous_distance_from_des_point
        #pitch_difference =  delta_pitch - self.previous_pitch
        height_difference = delta_height - self.previous_height
        reward = 0

        is_leg_up = False
        is_good_decrease = False
        is_good_place = False
        is_reached_goal = False

        if not done:
            #if robot is in standing up position
            if(self.is_robot_front_leg_up()):
                print("FRONT LEG IS UP!!!")
                reward +=10 #10
                is_leg_up = True
            #if achiving goal distance and height is decreasing 
            if distance_difference < 0.00 and height_difference < 0.00: #0.0
                print("DECREASE IN DISTANCE GOOD" ) 
                reward += 5 #5
                is_good_decrease = True
            #if robot is in certain range
            if distance_from_des_point < 0.25 and delta_pitch < 0.3:
                print("GOOD PLACE")
                reward += 50 #50
                is_good_place = True
        else:
            #if goal is reached
            if self.is_in_desired_position(observations):
                reward = 3*self.end_episode_points
                print ("REACHED THE GOAL!!!")
                is_reached_goal = True
            else:
                reward = -1*self.end_episode_points
        
        self.visualize_reward(is_leg_up,is_good_decrease,is_good_place, is_reached_goal)
        self.previous_distance_from_des_point = distance_from_des_point
        self.previous_pitch = delta_pitch
        self.previous_height = delta_height

        self.cumulated_reward += reward
        #print("Cumulated_reward=" + str(self.cumulated_reward))

        return reward
        
   
    def is_in_desired_position(self,observations, epsilon=0.05):
        """
        Here it return True if the current position is similar to the desired poistion
        """
        is_in_desired_pos = False
        current_position = Point()
        current_position.x = observations[0]
        current_position.y =0
        current_position.z = observations[1]
        distance = self.get_distance_from_desired_point(current_position, self.desired_point)

        if(abs(distance)<0.22 and abs(observations[2]-self.desired_pitch)<0.2):
            is_in_desired_pos = True
        return is_in_desired_pos


    def get_distance_from_desired_point(self, pstart, p_end): 
        """
        Here we get distance from current position to  Given a Vector3 Goal 
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance
 
    def has_crashed(self,odometry):
        """
        Here we check if robot is crashed or not
        """
        robot_has_crashed = False
        (roll, pitch, yaw) =  self.quaternion_to_euler_angle (w=odometry.pose.pose.orientation.w,x=odometry.pose.pose.orientation.x,y=odometry.pose.pose.orientation.y,z=odometry.pose.pose.orientation.z)

        if(abs(roll)>0.5 or abs(yaw)>0.5 or pitch>0.3):
            
            robot_has_crashed = True
        
        return robot_has_crashed

    def is_robot_front_leg_up(self):
        """
        Here we check if robot is in stand-up position which is robot's front legs are up, rear legs are on the ground and base has certain pitch angle
        """
        is_leg_up = False
        low_state = self.get_low_state()    
        odometry = self.get_odom()
        (roll, pitch, yaw) =  self.quaternion_to_euler_angle (w=odometry.pose.pose.orientation.w,x=odometry.pose.pose.orientation.x,y=odometry.pose.pose.orientation.y,z=odometry.pose.pose.orientation.z)

        #print("foot force: "+ str(low_state.footForce[0]) + " " + str(low_state.footForce[1]))
        if(low_state.footForce[0] ==0.0 and low_state.footForce[1]==0.0 and low_state.footForce[2] !=0.0 and low_state.footForce[3] !=0.0 and pitch < -0.3 and odometry.pose.pose.position.z>0.3):
            is_leg_up = True
        return is_leg_up
    
    def quaternion_to_euler_angle(self,w, x, y, z):
        """
        Here we convert quarternion to euler
        """
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        yaw = math.atan2(t3, t4)
        #print("roll" +str(roll)+ "yaw" + str(yaw)+"pitch" +str(pitch))
        return roll, pitch, yaw

    def euler_to_quaternion(self,yaw, pitch, roll):
        """
        Here we convert euler to quart
        """
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)

        return [qx, qy, qz, qw]

    def visualize_reward(self, is_leg_up, is_good_decrease, is_good_place, is_reached_goal):
        """
        Here we visualize reward, goal, and succes in Rviz for debugging
        """
        markerArray2 = MarkerArray()
        if(is_leg_up):
            marker = Marker()
            marker.header.frame_id = "/odom"
            marker.type = marker.CUBE
            marker.action = marker.ADD
            marker.id = 2
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            
            marker.pose.position.x = 0.5
            marker.pose.position.y = 0
            marker.pose.position.z = 1
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.lifetime = rospy.Duration.from_sec(0.5)

            markerArray2.markers.append(marker)
        
        if(is_good_decrease):
            marker = Marker()
            marker.header.frame_id = "/odom"
            marker.type = marker.CUBE
            marker.action = marker.ADD
            marker.id = 3
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0
            marker.color.g = 0
            marker.color.b = 1
            
            marker.pose.position.x = 0.5
            marker.pose.position.y = 0
            marker.pose.position.z = 1.5
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.lifetime = rospy.Duration.from_sec(0.5)
            markerArray2.markers.append(marker)
        if(is_good_place):
            marker = Marker()
            marker.header.frame_id = "/odom"
            marker.type = marker.CUBE
            marker.action = marker.ADD
            marker.id = 4
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            
            marker.pose.position.x = 0.5
            marker.pose.position.y = 0
            marker.pose.position.z = 2
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1            
            marker.lifetime = rospy.Duration.from_sec(0.5)
            markerArray2.markers.append(marker)

        if(is_reached_goal):
            markerArray = MarkerArray()
            marker = Marker()
            marker.header.frame_id = "/odom"
            marker.type = marker.ARROW
            marker.action = marker.ADD
            marker.id = 1
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            
            marker.pose.position.x = self.desired_point.x
            marker.pose.position.y = self.desired_point.y
            marker.pose.position.z = self.desired_point.z 
            q=self.euler_to_quaternion(0,self.desired_pitch,0)
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            markerArray2.markers.append(marker)    
        self.pub.publish(markerArray2)