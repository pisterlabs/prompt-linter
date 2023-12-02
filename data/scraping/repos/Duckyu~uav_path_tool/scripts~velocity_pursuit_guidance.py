#!/usr/bin/env python

import rospy
import sys
from os import path
from path_loader_class import pathLoader
from guidance_class import flightControl

if __name__ == '__main__':
    rospy.init_node('sitl_vel_pursuit_guide', anonymous=True)
    
    rate = rospy.Rate(60)
    
    file_dir = rospy.get_param("/file_dir")
    robot_name = rospy.get_param("robot_name")

    pl = pathLoader(file_dir, robot_name)
    print(pl)
    path = pl.load()
    pl.generate_path_msg(path)
    
    vel = rospy.get_param("vel")
    yaw_rate = rospy.get_param("yaw_rate")
    arv_dist = rospy.get_param("arv_dist")
    arv_yaw = rospy.get_param("arv_yaw")
    r_los = rospy.get_param("LoS_radius")

    rospy.loginfo("control  type: {0}".format(pl.ctrl_type))
    rospy.loginfo("rotation type: {0}".format(pl.rot_type))
    
    fc = flightControl(robot_name, [0,0,0,pl.start_pose[3]], vel, arv_dist, arv_yaw, path, pl.ctrl_type, pl.rot_type,  pl.cmd_type, r_los, yaw_rate)
    
    fc.takeoff()

    while not rospy.is_shutdown():
        try:
            fc.path_follow()
            fc.pub_odom()
            pl.pub_path()
            pl.pub_start_tf()
            rate.sleep()
        except rospy.ROSInterruptException:
            rospy.loginfo('Node terminated by pressing Ctrl+C!')
            sys.exit(0)
