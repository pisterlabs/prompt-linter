#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray, Int32, String
from geometry_msgs.msg import Pose, PoseStamped
from vanttec_uuv.msg import GuidanceWaypoints
from usv_perception.msg import obj_detected, obj_detected_list
from nav_msgs.msg import Path

# Class Definition
class AutoNav:
    def __init__(self):
        self.ned_x = 0
        self.ned_y = 0
        self.yaw = 0
        self.objects_list = []
        self.activated = True
        self.state = -1
        self.distance = 0
        self.InitTime = rospy.Time.now().secs
        self.offset = .55 #camera to ins offset
        self.target_x = 0
        self.target_y = 0
        self.ned_alpha = 0
        self.choose_side = 'left'
        self.distance_away = 5
        self.waypoints = GuidanceWaypoints()
        self.uuv_path = Path()
       
        #Waypoint test instead of perception node


        # ROS Subscribers
        rospy.Subscriber("/uuv_simulation/dynamic_model/pose", Pose, self.ins_pose_callback)
        '''
        rospy.Subscriber("/usv_perception/yolo_zed/objects_detected", obj_detected_list, self.objs_callback)
        '''

        # ROS Publishers
        self.uuv_waypoints = rospy.Publisher("/uuv_guidance/guidance_controller/waypoints", GuidanceWaypoints, queue_size=10)
        self.uuv_path_pub = rospy.Publisher("/uuv_planning/motion_planning/desired_path", Path, queue_size=10)
        self.status_pub = rospy.Publisher("/mission/status", Int32, queue_size=10)
        self.test = rospy.Publisher("/mission/state", Int32, queue_size=10)

        #Waypoint test instead of perception node

        self.objects_list = [
            {
                'X': 7,
                'Y': -4,
                'Z': 0
            },
            {
                'X': 7,
                'Y': 0,
                'Z': 0
            },
            {
                'X': 7,
                'Y': 4,
                'Z': 0                
            }            
        ]
    
    def ins_pose_callback(self,pose):
        self.ned_x = pose.position.x
        self.ned_y = pose.position.y
        self.ned_z = pose.position.z
        self.yaw = pose.orientation.z
    '''
    def objs_callback(self,data):
        self.objects_list = []
        for i in range(data.len):
            if str(data.objects[i].clase) == 'bouy':
                self.objects_list.append({'X' : data.objects[i].X + self.offset, 
                                      'Y' : data.objects[i].Y, 
                                      'color' : data.objects[i].color, 
                                      'class' : data.objects[i].clase})
    '''                                      

    def center_point(self):
        '''
        @name: center_point
        @brief: Returns two waypoints as desired positions. The first waypoint is
          between the middle of the gate and it right or left post, and the second a distance to the front 
        @param: --
        @return: --
        '''
        x_list = []
        y_list = []
        distance_list = []
        for i in range(len(self.objects_list)):
            x_list.append(self.objects_list[i]['X'])
            y_list.append(self.objects_list[i]['Y'])
            distance_list.append(math.pow(x_list[i]**2 + y_list[i]**2, 0.5))

        ind_g1 = np.argsort(distance_list)[0]
        ind_g2 = np.argsort(distance_list)[1]
        ind_g2 = np.argsort(distance_list)[2]

        x1 = x_list[ind_g1]
        y1 = -1*y_list[ind_g1]
        x2 = x_list[ind_g2]
        y2 = -1*y_list[ind_g2]
        x3 = x_list[ind_g2]
        y3 = -1*y_list[ind_g2]
        if (self.choose_side == 'left'):
            xc = min([x1,x2]) + abs(x1 - x2)/2 - self.distance_away
            yc = min([y1,y2]) + abs(y1 - y2)/2
            if y1 < y2:
                yl = y1
                xl = x1
                yr = y2
                xr = x2
            else:
                yl = y2
                xl = x2
                yr = y1
                xr = x1
        else:
            xc = min([x2,x3]) + abs(x2 - x3)/2 - self.distance_away
            yc = min([y2,y3]) + abs(y2 - y3)/2
            if y2 < y3:
                yl = y2
                xl = x2
                yr = y3
                xr = x3
            else:
                yl = y3
                xl = x3
                yr = y2
                xr = x2

        yd = yl - yr
        xd = xl - xr

        alpha = math.atan2(yd,xd) + math.pi/2
        if (abs(alpha) > (math.pi)):
            alpha = (alpha/abs(alpha))*(abs(alpha) - 2*math.pi)

        self.ned_alpha = alpha + self.yaw
        if (abs(self.ned_alpha) > (math.pi)):
            self.ned_alpha = (self.ned_alpha/abs(self.ned_alpha))*(abs(self.ned_alpha) - 2*math.pi)

        xm, ym = self.gate_to_body(3,0,alpha,xc,yc)

        self.target_x, self.target_y = self.body_to_ned(xm, ym)
        
        #path_array = Float32MultiArray()
        #path_array.layout.data_offset = 5
        #path_array.data = [xc, yc, xm, ym, 2]

        #self.desired(path_array)
        self.waypoints.guidance_law = 1
        self.waypoints.waypoint_list_length = 2
        self.waypoints.waypoint_list_x = [xc, xm]
        self.waypoints.waypoint_list_y = [yc, ym]
        self.waypoints.waypoint_list_z = [0,0]   
        self.desired(self.waypoints)

    def calculate_distance_to_sub(self):
        '''
        @name: calculate_distance_to_sub
        @brief: Returns the distance from the UUV to the next gate
        @param: --
        @return: --
        '''
        x_list = []
        y_list = []
        distance_list = []
        for i in range(len(self.objects_list)):
            x_list.append(self.objects_list[i]['X'])
            y_list.append(self.objects_list[i]['Y'])
            distance_list.append(math.pow(x_list[i]**2 + y_list[i]**2, 0.5))

        ind_g1 = np.argsort(distance_list)[0]
        ind_g2 = np.argsort(distance_list)[1]

        x1 = x_list[ind_g1]
        y1 = -1*y_list[ind_g1]
        x2 = x_list[ind_g2]
        y2 = -1*y_list[ind_g2]
        x3 = x_list[ind_g2]
        y3 = -1*y_list[ind_g2]

        if (self.choose_side == 'left'):
            xc = min([x1,x2]) + abs(x1 - x2)/2 
            yc = min([y1,y2]) + abs(y1 - y2)/2
            if y1 < y2:
                yl = y1
                xl = x1
                yr = y2
                xr = x2
            else:
                yl = y2
                xl = x2
                yr = y1
                xr = x1
        else:
            xc = min([x2,x3]) + abs(x2 - x3)/2
            yc = min([y2,y3]) + abs(y2 - y3)/2
            if y2 < y3:
                yl = y2
                xl = x2
                yr = y3
                xr = x3
            else:
                yl = y3
                xl = x3
                yr = y2
                xr = x2

        self.distance = math.pow(xc*xc + yc*yc, 0.5)

    def farther(self):
        '''
        @name: farther
        @brief: Returns a waypoint farther to the front of the vehicle in the NED
          reference frame to avoid perturbations.
        @param: --
        @return: --
        '''
        self.target_x, self.target_y = self.gate_to_ned(10, 0, 
                                                        self.ned_alpha,
                                                        self.target_x,
                                                        self.target_y)
        #path_array = Float32MultiArray()
        #path_array.layout.data_offset = 3
        #path_array.data = [self.target_x, self.target_y, 0]
        #self.desired(data)
        self.waypoints.guidance_law = 1
        self.waypoints.waypoint_list_length = 1
        self.waypoints.waypoint_list_x = {self.target_x}
        self.waypoints.waypoint_list_y = { self.target_y}
        self.waypoints.waypoint_list_z = {0}   
        self.desired(self.waypoints)

    def gate_to_body(self, gate_x2, gate_y2, alpha, body_x1, body_y1):
        '''
        @name: gate_to_body
        @brief: Coordinate transformation between gate and body reference frames.
        @param: gate_x2: target x coordinate in gate reference frame
                gate_y2: target y coordinate in gate reference frame
                alpha: angle between gate and body reference frames
                body_x1: gate x coordinate in body reference frame
                body_y1: gate y coordinate in body reference frame
        @return: body_x2: target x coordinate in body reference frame
                 body_y2: target y coordinate in body reference frame
        '''
        p = np.array([[gate_x2],[gate_y2]])
        J = self.rotation_matrix(alpha)
        n = J.dot(p)
        body_x2 = n[0] + body_x1
        body_y2 = n[1] + body_y1
        return (body_x2, body_y2)

    def body_to_ned(self, x2, y2):
        '''
        @name: body_to_ned
        @brief: Coordinate transformation between body and NED reference frames.
        @param: x2: target x coordinate in body reference frame
                y2: target y coordinate in body reference frame
        @return: ned_x2: target x coordinate in ned reference frame
                 ned_y2: target y coordinate in ned reference frame
        '''
        p = np.array([x2, y2])
        J = self.rotation_matrix(self.yaw)
        n = J.dot(p)
        ned_x2 = n[0] + self.ned_x
        ned_y2 = n[1] + self.ned_y
        return (ned_x2, ned_y2)

    def gate_to_ned(self, gate_x2, gate_y2, alpha, ned_x1, ned_y1):
        '''
        @name: gate_to_ned
        @brief: Coordinate transformation between gate and NED reference frames.
        @param: gate_x2: target x coordinate in gate reference frame
                gate_y2: target y coordinate in gate reference frame
                alpha: angle between gate and ned reference frames
                body_x1: gate x coordinate in ned reference frame
                body_y1: gate y coordinate in ned reference frame
        @return: body_x2: target x coordinate in ned reference frame
                 body_y2: target y coordinate in ned reference frame
        '''
        p = np.array([[gate_x2],[gate_y2]])
        J = self.rotation_matrix(alpha)
        n = J.dot(p)
        ned_x2 = n[0] + ned_x1
        ned_y2 = n[1] + ned_y1
        return (ned_x2, ned_y2)

    def rotation_matrix(self, angle):
        '''
        @name: rotation_matrix
        @brief: Transformation matrix template.
        @param: angle: angle of rotation
        @return: J: transformation matrix
        '''
        J = np.array([[math.cos(angle), -1*math.sin(angle)],
                      [math.sin(angle), math.cos(angle)]])
        return (J)

    def desired(self, path):
    	self.uuv_waypoints.publish(path)
        self.uuv_path.header.stamp = rospy.Time.now()
        self.uuv_path.header.frame_id = "world"
        del self.uuv_path.poses[:]
        for index in range(path.waypoint_list_length):
            pose = PoseStamped()
            pose.header.stamp       = rospy.Time.now()
            pose.header.frame_id    = "world"
            pose.pose.position.x    = path.waypoint_list_x[index]
            pose.pose.position.y    = path.waypoint_list_y[index]
            pose.pose.position.z    = path.waypoint_list_z[index]
            self.uuv_path.poses.append(pose)
        self.uuv_path_pub.publish(self.uuv_path)
def main():
    rospy.init_node("auto_nav_position", anonymous=False)
    rate = rospy.Rate(20)
    autoNav = AutoNav()
    autoNav.distance = 4
    last_detection = []
    while not rospy.is_shutdown() and autoNav.activated:
        rospy.loginfo("AutoNav is activated")
        #rospy.loginfo(autoNav.objects_list)
        rospy.loginfo(last_detection)
        if autoNav.objects_list != last_detection:
            rospy.loginfo("Last detection not activated")
            if autoNav.state == -1:
                rospy.loginfo("AutoNav.state == -1")
                while (not rospy.is_shutdown()) and (len(autoNav.objects_list) < 3):
                    autoNav.test.publish(autoNav.state)
                    rospy.loginfo("AutoNav.state in -1")
                    rate.sleep()
                autoNav.state = 0
               # last_detection = autoNav.objects_list

            if autoNav.state == 0:
                rospy.loginfo("AutoNav.state == 0")
                autoNav.test.publish(autoNav.state)
                if len(autoNav.objects_list) >= 3:
                    rospy.loginfo("AutoNav.objects_list) >= 3")
                    autoNav.calculate_distance_to_sub()
                if (len(autoNav.objects_list) >= 3) and (autoNav.distance >= 2):
                    rospy.loginfo("AutoNav.objects_list) >= 3 and (autoNav.distance >= 2)")
                    autoNav.center_point()
                else:
                    rospy.loginfo("No autoNav.objects_list")
                    initTime = rospy.Time.now().secs
                    while ((not rospy.is_shutdown()) and 
                        (len(autoNav.objects_list) < 3 or autoNav.distance < 2)):
                        rospy.loginfo("not rospy.is_shutdown() and  (len(autoNav.objects_list) < 3 or autoNav.distance < 2)")
                        if rospy.Time.now().secs - initTime > 2:
                            rospy.loginfo("rospy.Time.now().secs - initTime > 2")
                            autoNav.state = 1
                            rate.sleep()
                            break
                #last_detection = autoNav.objects_list

        if autoNav.state == 1:
            rospy.loginfo("AutoNav.state == 1")
            autoNav.test.publish(autoNav.state)
            if len(autoNav.objects_list) >= 3:
                autoNav.state = 2
            else:
                initTime = rospy.Time.now().secs
                while ((not rospy.is_shutdown()) and 
                    (len(autoNav.objects_list) < 3)):
                    if rospy.Time.now().secs - initTime > 1:
                        autoNav.farther()
                        rate.sleep()
                        break
            #last_detection = autoNav.objects_list

        if autoNav.objects_list != last_detection:
            rospy.loginfo("autoNav.objects_list != last_detection:")
            if autoNav.state == 2:
                rospy.loginfo("AutoNav.state == 2")
                autoNav.test.publish(autoNav.state)
                if len(autoNav.objects_list) >= 3:
                    autoNav.calculate_distance_to_sub()
                if len(autoNav.objects_list) >= 3 and autoNav.distance >= 2:
                    autoNav.center_point()
                else:
                    initTime = rospy.Time.now().secs
                    while ((not rospy.is_shutdown()) and 
                        (len(autoNav.objects_list) < 3 or autoNav.distance < 2)):
                        if rospy.Time.now().secs - initTime > 2:
                            autoNav.state = 3
                            rate.sleep()
                            break
               # last_detection = autoNav.objects_list

        elif autoNav.state == 3:
            autoNav.test.publish(autoNav.state)
            time.sleep(1)
            autoNav.status_pub.publish(1)

        rate.sleep()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
