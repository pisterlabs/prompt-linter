#! /usr/bin/env python


from theconstruct_msgs.msg import RLExperimentInfo
import numpy
import rospy
from std_msgs.msg import Float64
from abb_catkin.srv import JointTraj, JointTrajRequest, EeRpy, EeRpyRequest
from openai_ros import ABB_client


class Abbenv(ABB_client.RobotGazeboEnv):
    """Superclass for all Fetch environments.
    """

    def zz(self):

        # MyControllers=['arm_controller', 'joint_state_controller']
        #
        # switch_controller = rospy.ServiceProxy('controller_manager/switch_controller', SwitchController)
        # list_controllers = rospy.ServiceProxy('controller_manager/list_controllers', ListControllers)
        #
        # switch_controller.wait_for_service()
        # list_controllers.wait_for_service()

        print ("Entered ABB Env")
        super(Abbenv, self).set_trajectory_joints([ 0, 1, 0.5, 0, 0.3, 0.2 ])

if __name__ == "__main__":

    rospy.init_node("unspawner", anonymous=True)
    Abbenv().zz()
   # rospy.spin()

#    print(ABB_client.RobotGazeboEnv.set_trajectory_joints([ 0, 0, 0, 0, 0, 0 ]))
#    print(get_ee_rpy())
