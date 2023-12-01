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
        print ("Entered ABB Env")
        super(Abbenv, self).set_trajectory_joints([ 1, 0.3, -0.3, 0, 1, -1 ])

if __name__ == "__main__":

     Abbenv().zz()

#    print(ABB_client.RobotGazeboEnv.set_trajectory_joints([ 0, 0, 0, 0, 0, 0 ]))
#    print(get_ee_rpy())
