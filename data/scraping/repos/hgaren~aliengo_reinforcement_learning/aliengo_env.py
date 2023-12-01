
'''
Openai-ROS Aliengo Robot Environment
Includes ROS Related Gazebo Connection and Aliengo Controller  which connects to  Open-ai Env

Garen Haddeler
12.11.2020

'''

import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Twist, WrenchStamped
from openai_ros.openai_ros_common import ROSLauncher
import time
from laikago_msgs.msg import LowCmd,LowState,MotorState,MotorCmd

class AliengoEnv(robot_gazebo_env.RobotGazeboEnv):
     
    def __init__(self, ros_ws_abspath):
      
        print("Start AliengoEnv INIT...")
      
        # We launch the ROSlaunch that spawns the robot into the world
        ROSLauncher(rospackage_name="aliengo_gazebo",
                    launch_file_name="aliengo_empty_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []
        self.lowState = LowState()
        self.motorState = MotorState()
        self.lowCmd = LowCmd()
        self.odom = Odometry()
        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
      
        super(AliengoEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")
       



        self.gazebo.unpauseSim()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/aliengo/odometry", Odometry, self.odomCallback)
        rospy.Subscriber("/laikago_gazebo/FR_hip_controller/state", MotorState, self.FRhipCallback)
        rospy.Subscriber("/laikago_gazebo/FR_thigh_controller/state", MotorState, self.FRthighCallback)
        rospy.Subscriber("/laikago_gazebo/FR_calf_controller/state", MotorState, self.FRcalfCallback)
        rospy.Subscriber("/laikago_gazebo/FL_hip_controller/state", MotorState, self.FLhipCallback)
        rospy.Subscriber("/laikago_gazebo/FL_thigh_controller/state", MotorState, self.FLthighCallback)
        rospy.Subscriber("/laikago_gazebo/FL_calf_controller/state", MotorState, self.FLcalfCallback)
        rospy.Subscriber("/laikago_gazebo/RR_hip_controller/state", MotorState, self.RRhipCallback)
        rospy.Subscriber("/laikago_gazebo/RR_thigh_controller/state", MotorState, self.RRthighCallback)
        rospy.Subscriber("/laikago_gazebo/RR_calf_controller/state",  MotorState, self.RRcalfCallback)
        rospy.Subscriber("/laikago_gazebo/RL_hip_controller/state", MotorState, self.RLhipCallback)
        rospy.Subscriber("/laikago_gazebo/RL_thigh_controller/state", MotorState, self.RLthighCallback)
        rospy.Subscriber("/laikago_gazebo/RL_calf_controller/state", MotorState, self.RLcalfCallback)
        
        rospy.Subscriber("/visual/FR_foot_contact/the_force", WrenchStamped, self.FRfootCallback)
        rospy.Subscriber("/visual/FL_foot_contact/the_force", WrenchStamped, self.FLfootCallback)
        rospy.Subscriber("/visual/RR_foot_contact/the_force", WrenchStamped, self.RRfootCallback)
        rospy.Subscriber("/visual/RL_foot_contact/the_force", WrenchStamped, self.RLfootCallback)


        self.lowState_pub = rospy.Publisher("/laikago_gazebo/lowState/state", LowState, queue_size=1)
        self.servo_pub_0 = rospy.Publisher("/laikago_gazebo/FR_hip_controller/command", MotorCmd, queue_size=1)
        self.servo_pub_1 = rospy.Publisher("/laikago_gazebo/FR_thigh_controller/command", MotorCmd, queue_size=1)
        self.servo_pub_2 = rospy.Publisher("/laikago_gazebo/FR_calf_controller/command", MotorCmd, queue_size=1)
        self.servo_pub_3 = rospy.Publisher("/laikago_gazebo/FL_hip_controller/command",  MotorCmd, queue_size=1)
        self.servo_pub_4 = rospy.Publisher("/laikago_gazebo/FL_thigh_controller/command", MotorCmd, queue_size=1)
        self.servo_pub_5 = rospy.Publisher("/laikago_gazebo/FL_calf_controller/command",  MotorCmd, queue_size=1)
        self.servo_pub_6 = rospy.Publisher("/laikago_gazebo/RR_hip_controller/command",  MotorCmd, queue_size=1)
        self.servo_pub_7 = rospy.Publisher("/laikago_gazebo/RR_thigh_controller/command",  MotorCmd, queue_size=1)
        self.servo_pub_8 = rospy.Publisher("/laikago_gazebo/RR_calf_controller/command",  MotorCmd, queue_size=1)
        self.servo_pub_9 = rospy.Publisher("/laikago_gazebo/RL_hip_controller/command",  MotorCmd, queue_size=1)
        self.servo_pub_10 = rospy.Publisher("/laikago_gazebo/RL_thigh_controller/command",  MotorCmd, queue_size=1)
        self.servo_pub_11 = rospy.Publisher("/laikago_gazebo/RL_calf_controller/command",  MotorCmd, queue_size=1)

        
        self.gazebo.pauseSim()
        

        rospy.logdebug("Finished AliengoEnv INIT...")

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        return True
    def odomCallback(self, msg):
        self.odom = msg

    def FRhipCallback(self, msg):
        self.lowState.motorState[0].mode = msg.mode
        self.lowState.motorState[0].position = msg.position
        self.lowState.motorState[0].velocity = msg.velocity
        self.lowState.motorState[0].torque = msg.torque


    def FRthighCallback(self, msg):
        self.lowState.motorState[1].mode = msg.mode 
        self.lowState.motorState[1].position = msg.position 
        self.lowState.motorState[1].velocity = msg.velocity 
        self.lowState.motorState[1].torque = msg.torque 
     

    def FRcalfCallback(self, msg):
        self.lowState.motorState[2].mode = msg.mode 
        self.lowState.motorState[2].position = msg.position 
        self.lowState.motorState[2].velocity = msg.velocity 
        self.lowState.motorState[2].torque = msg.torque 
     

    def FLhipCallback(self, msg):
        self.lowState.motorState[3].mode = msg.mode 
        self.lowState.motorState[3].position = msg.position 
        self.lowState.motorState[3].velocity = msg.velocity 
        self.lowState.motorState[3].torque = msg.torque 
     

    def FLthighCallback(self, msg):
        self.lowState.motorState[4].mode = msg.mode 
        self.lowState.motorState[4].position = msg.position 
        self.lowState.motorState[4].velocity = msg.velocity 
        self.lowState.motorState[4].torque = msg.torque 
     

    def FLcalfCallback(self, msg):
        self.lowState.motorState[5].mode = msg.mode 
        self.lowState.motorState[5].position = msg.position 
        self.lowState.motorState[5].velocity = msg.velocity 
        self.lowState.motorState[5].torque = msg.torque 

    def RRhipCallback(self, msg):
        self.lowState.motorState[6].mode = msg.mode 
        self.lowState.motorState[6].position = msg.position 
        self.lowState.motorState[6].velocity = msg.velocity 
        self.lowState.motorState[6].torque = msg.torque 

    def RRthighCallback(self, msg):
        self.lowState.motorState[7].mode = msg.mode 
        self.lowState.motorState[7].position = msg.position 
        self.lowState.motorState[7].velocity = msg.velocity 
        self.lowState.motorState[7].torque = msg.torque 
     

    def RRcalfCallback(self, msg):
        self.lowState.motorState[8].mode = msg.mode 
        self.lowState.motorState[8].position = msg.position 
        self.lowState.motorState[8].velocity = msg.velocity 
        self.lowState.motorState[8].torque = msg.torque 
     

    def RLhipCallback(self, msg):
        self.lowState.motorState[9].mode = msg.mode 
        self.lowState.motorState[9].position = msg.position 
        self.lowState.motorState[9].velocity = msg.velocity 
        self.lowState.motorState[9].torque = msg.torque 
     

    def RLthighCallback(self, msg):
        self.lowState.motorState[10].mode = msg.mode 
        self.lowState.motorState[10].position = msg.position 
        self.lowState.motorState[10].velocity = msg.velocity 
        self.lowState.motorState[10].torque = msg.torque 
     

    def RLcalfCallback(self, msg):
        self.lowState.motorState[11].mode = msg.mode 
        self.lowState.motorState[11].position = msg.position 
        self.lowState.motorState[11].velocity = msg.velocity 
        self.lowState.motorState[11].torque = msg.torque 
     
    def FRfootCallback(self, msg):
        self.lowState.eeForce[0].x = msg.wrench.force.x;
        self.lowState.eeForce[0].y = msg.wrench.force.y;
        self.lowState.eeForce[0].z = msg.wrench.force.z;
        self.lowState.footForce[0] = msg.wrench.force.z;
    

    def FLfootCallback(self, msg):
        self.lowState.eeForce[1].x = msg.wrench.force.x;
        self.lowState.eeForce[1].y = msg.wrench.force.y;
        self.lowState.eeForce[1].z = msg.wrench.force.z;
        self.lowState.footForce[1] = msg.wrench.force.z;
    

    def RRfootCallback(self, msg):
        self.lowState.eeForce[2].x = msg.wrench.force.x;
        self.lowState.eeForce[2].y = msg.wrench.force.y;
        self.lowState.eeForce[2].z = msg.wrench.force.z;
        self.lowState.footForce[2] = msg.wrench.force.z;
    

    def RLfootCallback(self, msg):
    
        self.lowState.eeForce[3].x = msg.wrench.force.x;
        self.lowState.eeForce[3].y = msg.wrench.force.y;
        self.lowState.eeForce[3].z = msg.wrench.force.z;
        self.lowState.footForce[3] = msg.wrench.force.z;
    
    def paramInit(self):
        for i in range(0,4):
            self.lowCmd.motorCmd[i*3+0].mode = 0x0A 
            self.lowCmd.motorCmd[i*3+0].positionStiffness = 70 
            self.lowCmd.motorCmd[i*3+0].velocity = 0 
            self.lowCmd.motorCmd[i*3+0].velocityStiffness = 3 
            self.lowCmd.motorCmd[i*3+0].torque = 0 
            self.lowCmd.motorCmd[i*3+1].mode = 0x0A 
            self.lowCmd.motorCmd[i*3+1].positionStiffness = 180 
            self.lowCmd.motorCmd[i*3+1].velocity = 0 
            self.lowCmd.motorCmd[i*3+1].velocityStiffness = 8 
            self.lowCmd.motorCmd[i*3+1].torque = 0 
            self.lowCmd.motorCmd[i*3+2].mode = 0x0A 
            self.lowCmd.motorCmd[i*3+2].positionStiffness = 300 
            self.lowCmd.motorCmd[i*3+2].velocity = 0 
            self.lowCmd.motorCmd[i*3+2].velocityStiffness = 15 
            self.lowCmd.motorCmd[i*3+2].torque = 0 
         
        for i in range(0,4):
            self.lowCmd.motorCmd[i].position =  self.lowState.motorState[i].position 

    def sendServoCmd(self):
        
    
        self.servo_pub_0.publish(self.lowCmd.motorCmd[0])
        self.servo_pub_1.publish(self.lowCmd.motorCmd[1])
        self.servo_pub_2.publish(self.lowCmd.motorCmd[2])
        self.servo_pub_3.publish(self.lowCmd.motorCmd[3])
        self.servo_pub_4.publish(self.lowCmd.motorCmd[4])
        self.servo_pub_5.publish(self.lowCmd.motorCmd[5])
        self.servo_pub_6.publish(self.lowCmd.motorCmd[6])
        self.servo_pub_7.publish(self.lowCmd.motorCmd[7])
        self.servo_pub_8.publish(self.lowCmd.motorCmd[8])
        self.servo_pub_9.publish(self.lowCmd.motorCmd[9])
        self.servo_pub_10.publish(self.lowCmd.motorCmd[10])
        self.servo_pub_11.publish(self.lowCmd.motorCmd[11])
        #self.wait_time_for_execute_movement( self.lowState.motorState,self.lowCmd.motorCmd,0.3)
    

    def check_array_similar(self, ref_value_array, check_value_array, epsilon):
        """
        It checks if the check_value id similar to the ref_value
        """
        return numpy.allclose(ref_value_array, check_value_array, atol=epsilon)
    def moveAllPosition(self,targetPos):
        percent = 0.0
        while percent < 1:
            for m in range(0,12):
                self.lowCmd.motorCmd[m].position =self.lowState.motorState[m].position*(1-percent)+ percent*targetPos[m]
            self.sendServoCmd()
            percent = percent +0.1
            time.sleep(0.01)    
         

    def get_odom(self):
        return self.odom

    def get_low_state(self):
        return self.lowState