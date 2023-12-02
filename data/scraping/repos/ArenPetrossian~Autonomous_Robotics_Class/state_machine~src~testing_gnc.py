#! /usr/bin/env python
import rospy
from computer_vision.msg import target
from guidance_navigation_control.msg import sensorInfo_actuatorStatus
from guidance_navigation_control.msg import task_desiredAction
from guidance_navigation_control.msg import controlCommand
from time import sleep

def cv_data(cv_data):
	return
        #print (cv_data)

def sensor_data(sensor_data):
	return
	#print (sensor_data)

def gnc_feedback(gnc_feedback):
	return
	#print (gnc_feedback)

while True:
	rospy.init_node('SMACH')
	smach_pub = rospy.Publisher('task_desiredAction', task_desiredAction, queue_size=10)
	rospy.Subscriber('target', target, cv_data)
	rospy.Subscriber('sensorInfo_actuatorStatus', sensorInfo_actuatorStatus, sensor_data)
        rospy.Subscriber('controlCommand', controlCommand, gnc_feedback)
	rate = rospy.Rate(1)
	final_message = task_desiredAction()
	while not rospy.is_shutdown():
		final_message.yaw_set = 14
                final_message.distance_set = 5
		final_message.depth_set = 20
		userdata = input("Enter 0 for False or 1 for True:")
		if userdata == 0:
			final_message.bumpIntoBuoy = False
		else:
			final_message.surface = True
			final_message.bumpIntoBuoy = True
		smach_pub.publish(final_message)
		final_message.bumpIntoBuoy = False
		sleep(2)
		smach_pub.publish(final_message)
		rate.sleep()

	else:
		exit()
