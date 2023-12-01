#! /usr/bin/env python
import rospy
from guidance_navigation_control.msg import task_desiredAction
from guidance_navigation_control.msg import sensorInfo_actuatorStatus

def task_data(task_data):
	print("State machine: ")
        print(task_data)

def sensor_actuator_data(sensor_actuator_data):
	print("Arduino: ")
	print (sensor_actuator_data)

while True:
	print("GUI")
	rospy.init_node('GUI')
	rospy.Subscriber('task_desiredAction', task_desiredAction, task_data)
	rospy.Subscriber('sensorInfo_actuatorStatus', sensorInfo_actuatorStatus, sensor_actuator_data)
	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		rate.sleep()

	else:
		exit()
