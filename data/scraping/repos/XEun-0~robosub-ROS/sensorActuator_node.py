#! /usr/bin/env python
import rospy
from guidance_navigation_control.msg import controlCommand
from guidance_navigation_control.msg import  sensorInfo_actuatorStatus

def gnc_data(gnc_data):
	print (gnc_data)

while True:
	rospy.init_node('SENSORS_ACTUATORS')
	sensorActuator_pub = rospy.Publisher('sensorInfo_actuatorStatus', sensorInfo_actuatorStatus, queue_size=10)
	rospy.Subscriber('controlCommand', controlCommand, gnc_data)
	rate = rospy.Rate(10)
	final_message = sensorInfo_actuatorStatus()
	while not rospy.is_shutdown():
		final_message.yaw_current = 17
                final_message.depth_current = 21
		final_message.temperature = 72
		final_message.thruster_values[0] = 1600
                final_message.thruster_values[1] = 1300
                final_message.thruster_values[2] = 1700
		final_message.thruster_values[3] = 1200
                final_message.thruster_values[4] = 1500
                final_message.thruster_values[5] = 1800
		final_message.stabilized = True
		sensorActuator_pub.publish(final_message)
		rate.sleep()

	else:
		exit()
