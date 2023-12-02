#! /usr/bin/env python
import rospy
from guidance_navigation_control.msg import task_desiredAction
from guidance_navigation_control.msg import controlCommand
from guidance_navigation_control.msg import sensorInfo_actuatorStatus

def smach_data(smach_data):
	print (smach_data)

def stabilized_data(stabilized_data):
        print (stabilized_data)

while True:
	rospy.init_node('GNC')
	gnc_pub = rospy.Publisher('controlCommand', controlCommand, queue_size=10)
	rospy.Subscriber('task_desiredAction', task_desiredAction, smach_data)
        rospy.Subscriber('sensorInfo_actuatorStatus', sensorInfo_actuatorStatus, stabilized_data)
	rate = rospy.Rate(10)
	final_message = controlCommand()
	while not rospy.is_shutdown():
		final_message.yaw_set = 60
                final_message.pitch_set = 30
		final_message.depth_set = 1
		final_message.final_command = True
		gnc_pub.publish(final_message)
		rate.sleep()

	else:
		exit()
