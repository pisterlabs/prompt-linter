#!/usr/bin/env python3
import rospy
import numpy as np
from common_tools.lib import guidanceNodeInit, nodeEnd, ps4, tau, observer, reference, s_p
from common_tools.math_tools import *
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, Float64

### Custom code here ###




if __name__ == '__main__':

    node = guidanceNodeInit()
    r = rospy.Rate(100)

    while not rospy.is_shutdown():
        ### Function and method calls here ###
        r.sleep()
    
    nodeEnd(node)
