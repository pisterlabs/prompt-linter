from openai_ros import robot_gazebo_env
import numpy as np
import rospy
import time
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty


class NaoRobotEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all Robot environments.
    """

    def __init__(self):
        """Initializes a new Robot environment.
        """
        # Variables that we give through the constructor.

        # Internal Vars
        # initialize all the rospy publishers here for all the joints in the nao 2+4+8+6+2+4 = 26

        ############ HEAD (2) ##############
        self.head_pitch = rospy.Publisher('/nao_dcm/HeadPitch_position_controller/command', Float64, queue_size=1)
        self.head_yaw = rospy.Publisher('/nao_dcm/HeadYaw_position_controller/command', Float64, queue_size=1)
        ############ SHOULDER (4) ############
        self.l_shoulder_pitch = rospy.Publisher('/nao_dcm/LShoulderPitch_position_controller/command', Float64,
                                                queue_size=1)

        self.l_shoulder_roll = rospy.Publisher('/nao_dcm/LShoulderRoll_position_controller/command', Float64,
                                               queue_size=1)
        self.r_shoulder_pitch = rospy.Publisher('/nao_dcm/RShoulderPitch_position_controller/command', Float64,
                                                queue_size=1)

        self.r_shoulder_roll = rospy.Publisher('/nao_dcm/RShoulderRoll_position_controller/command', Float64,
                                               queue_size=1)
        ############# HAND (8) ################
        self.l_elbow_roll = rospy.Publisher('/nao_dcm/LElbowRoll_position_controller/command', Float64, queue_size=1)
        self.l_elbow_yaw = rospy.Publisher('/nao_dcm/LElbowYaw_position_controller/command', Float64, queue_size=1)
        self.r_elbow_roll = rospy.Publisher('/nao_dcm/RElbowRoll_position_controller/command', Float64, queue_size=1)
        self.r_elbow_yaw = rospy.Publisher('/nao_dcm/RElbowYaw_position_controller/command', Float64, queue_size=1)
        self.l_hand_pos = rospy.Publisher('/nao_dcm/LHand_position_controller/command', Float64, queue_size=1)
        self.r_hand_pos = rospy.Publisher('/nao_dcm/RHand_position_controller/command', Float64, queue_size=1)
        self.l_wrist_yaw = rospy.Publisher('/nao_dcm/LWristYaw_position_controller/command', Float64, queue_size=1)
        self.r_wrist_yaw = rospy.Publisher('/nao_dcm/RWristYaw_position_controller/command', Float64, queue_size=1)

        ############# HIP (6) ################
        self.l_hip_pitch = rospy.Publisher('/nao_dcm/LHipPitch_position_controller/command', Float64, queue_size=1)
        self.l_hip_roll = rospy.Publisher('/nao_dcm/LHipRoll_position_controller/command', Float64, queue_size=1)
        self.l_hip_yaw_pitch = rospy.Publisher('/nao_dcm/LHipYawPitch_position_controller/command', Float64,
                                               queue_size=1)
        self.r_hip_pitch = rospy.Publisher('/nao_dcm/RHipPitch_position_controller/command', Float64, queue_size=1)

        self.r_hip_roll = rospy.Publisher('/nao_dcm/RHipRoll_position_controller/command', Float64, queue_size=1)
        self.r_hip_yaw_pitch = rospy.Publisher('/nao_dcm/RHipYawPitch_position_controller/command', Float64,
                                               queue_size=1)

        ############## KNEE (2) ##############
        self.l_knee_pitch = rospy.Publisher('/nao_dcm/LKneePitch_position_controller/command', Float64, queue_size=1)
        self.r_knee_pitch = rospy.Publisher('/nao_dcm/RKneePitch_position_controller/command', Float64, queue_size=1)

        ############## ANKLE (4) ############
        self.l_ankle_pitch = rospy.Publisher('/nao_dcm/LAnklePitch_position_controller/command', Float64, queue_size=1)
        self.l_ankle_roll = rospy.Publisher('/nao_dcm/LAnkleRoll_position_controller/command', Float64, queue_size=1)
        self.r_ankle_pitch = rospy.Publisher('/nao_dcm/RAnklePitch_position_controller/command', Float64, queue_size=1)
        self.r_ankle_roll = rospy.Publisher('/nao_dcm/RAnkleRoll_position_controller/command', Float64, queue_size=1)

        # self.controllers_list = ['HeadPitch','HeadYaw','LShoulderPitch','LShoulderRoll','RShoulderPitch','RShoulderRoll',\
        #                           'LElbowRoll','LElbowYaw','RElbowRoll','RElbowYaw','LHand','RHand','LWristYaw','RWristYaw',\
        #                           'LHipPitch','LHipRoll','LHipYawPitch','RHipPitch','RHipRoll','RHipYawPitch',\
        #                           'LKneePitch','RKneePitch','LAnklePitch','LAnkleRoll','RAnklePitch','RAnkleRoll']

        self.controllers_list = [ 'LHipPitch','LHipRoll','LHipYawPitch','RHipPitch','RHipRoll','RHipYawPitch',\
                                  'LKneePitch','RKneePitch','LAnklePitch','LAnkleRoll','RAnklePitch','RAnkleRoll']


        # self.publishers_array = [self.head_pitch,self.head_yaw,self.l_shoulder_pitch,self.l_shoulder_roll,self.r_shoulder_pitch,self.r_shoulder_roll,\
        #                           self.l_elbow_roll,self.l_elbow_yaw,self.r_elbow_roll,self.r_elbow_yaw,self.l_hand_pos,self.r_hand_pos,\
        #                           self.l_wrist_yaw,self.r_wrist_yaw,self.l_hip_pitch,self.l_hip_roll,self.l_hip_yaw_pitch,self.r_hip_pitch,\
        #                           self.r_hip_roll,self.r_hip_yaw_pitch,self.l_knee_pitch,self.r_knee_pitch,self.l_ankle_pitch,self.l_ankle_roll,\
        #                           self.r_ankle_pitch,self.r_ankle_roll]

        self.publishers_array = [self.l_hip_pitch,self.l_hip_roll,self.l_hip_yaw_pitch,self.r_hip_pitch,\
                                  self.r_hip_roll,self.r_hip_yaw_pitch,self.l_knee_pitch,self.r_knee_pitch,self.l_ankle_pitch,self.l_ankle_roll,\
                                  self.r_ankle_pitch,self.r_ankle_roll]

        # It doesnt use namespace
        self.robot_name_space = ""

        reset_controls_bool = True or False

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(NaoRobotEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")
        self.gazebo.unpauseSim()
        print('Unpausing and pausing gazebo ...')
        self.gazebo.pauseSim()

        rospy.logdebug("Finished NaoRobotEnv INIT...")


    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        # TODO
        return True

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

    def _move_robot(self, action, epsilon=0.05, update_rate=10, time_sleep=0.05, check_position=True):
        """Applies the given action to the simulation.
        """
        # build list of commands for the action space
        commands = [Float64()]*len(self.controllers_list)
        for i in range(len(commands)):
            # publish the commands
            commands[i].data = action[i]
            rospy.logdebug("NaoJointsPos>>" + str(commands[i]))
            self.publishers_array[i].publish(commands[i])

        # if check_position:
        #     self.wait_time_for_execute_movement(action, epsilon, update_rate)
        # else:
        #     self.wait_time_movement_hard(time_sleep=time_sleep)
        time.sleep(1)
        print('action completed')

    def wait_time_movement_hard(self, time_sleep):
        """
        Hard Wait to avoid inconsistencies in times executing actions
        """
        rospy.logdebug("Test Wait="+str(time_sleep))
        time.sleep(time_sleep)

    def wait_time_for_execute_movement(self, joints_array, epsilon, update_rate):
        """
        We wait until Joints are where we asked them to be based on the joints_states
        :param joints_array:Joints Values in radians of each of the three joints of hopper leg.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the joint_states.
        :return:
        """
        rospy.logdebug("START wait_until_joint_position_achieved...")

        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0

        rospy.logdebug("Desired JointsState>>" + str(joints_array))
        rospy.logdebug("epsilon>>" + str(epsilon))

        while not rospy.is_shutdown():
            current_joint_states = self._check_joint_states_ready()
            # values_to_check = list(current_joint_states.position)
            values_to_check = current_joint_states
            vel_values_are_close = self.check_array_similar(joints_array, values_to_check, epsilon)
            if vel_values_are_close:
                rospy.logdebug("Reached JointStates!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rospy.logdebug("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time - start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time) + "]")

        rospy.logdebug("END wait_until_jointstate_achieved...")

        return delta_time

    def check_array_similar(self, ref_value_array, check_value_array, epsilon):
        """
        It checks if the check_value id similar to the ref_value
        """
        print('reference value : ',ref_value_array)
        print('check value : ',check_value_array)
        rospy.logdebug("ref_value_array=" + str(ref_value_array))
        rospy.logdebug("check_value_array=" + str(check_value_array))
        print('current norm :: ',np.linalg.norm(np.array(ref_value_array)-np.array(check_value_array)))
        return np.allclose(ref_value_array, check_value_array, atol=epsilon)

    def _check_joint_states_ready(self):
        self.joint_states = None
        rospy.logdebug("Waiting for /joint_states to be READY...")
        while self.joint_states is None and not rospy.is_shutdown():
            try:
                self.joint_states = rospy.wait_for_message("/joint_states", JointState, timeout=10.0)
                rospy.logdebug("Current /joint_states READY=>")
            except:
                rospy.logerr("Current /joint_states not ready yet, retrying for getting joint_states")
        joint_states_in_order = self.find_index_return_joint_states(self.joint_states,self.controllers_list)
        return joint_states_in_order

    def find_index_return_joint_states(self,joint_states,controllers_list):
        """
        function that finds the names of joints from controllers_list in joint_states.name
        and returns them in the order as in controllers_list
        :param controllers_list: list containing names of the joints
        :return: joint values of the joints in the same order as in controllers_list
        """
        names_list = joint_states.name
        return_array_ = []
        for name in controllers_list:
            if name in names_list:
                pos = names_list.index(name)
                return_array_.append(joint_states.position[pos])
            else:
                rospy.logerr("Something wrong with simulation. Could not find the right joint from sensor measurements.")
                exit(0)
        return return_array_

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
    # Methods that the TrainingEnvironment will need.
    # ----------------------------