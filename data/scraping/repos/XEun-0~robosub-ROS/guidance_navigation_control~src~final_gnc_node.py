#! /usr/bin/env python
import rospy
from guidance_navigation_control.msg import task_desiredAction
from guidance_navigation_control.msg import controlCommand
from guidance_navigation_control.msg import sensorInfo_actuatorStatus
from Subscriber import Subscribe_to
from time import sleep


class Guidance_Navigation_Control():
	def __init__(self):
		print("starting")
		self.smach_sub = Subscribe_to('task_desiredAction')
		self.sensors_sub = Subscribe_to('sensorInfo_actuatorStatus')
		self.setpoints = controlCommand()


	def data_received(self):
		if (self.sensors_sub.was_data_sent()):
			if (self.smach_sub.was_data_sent()):
				return True
			else:
				self.setpoints.distance_set = 0
				gnc_pub.publish(self.setpoints)
				return False
		return False


	def main_loop(self):
		print("looping")
		if self.smach_sub.was_data_sent():
			print("data_updated")
			self.smach_data = self.smach_sub.get_data()
			self.sensors_data = self.sensors_sub.get_data()
			if (self.smach_data.surface):
				self.surface()
			elif (self.smach_data.bumpIntoBuoy):
				self.bumpIntoBuoy()
			self.update_setpoints()
			gnc_pub.publish(self.setpoints)


	def update_setpoints(self):
		print("updating")
		self.setpoints.yaw_set = self.smach_data.yaw_set + self.sensors_data.yaw_current
		if (self.setpoints.yaw_set > 360):
			self.setpoints.yaw_set -= 360
		elif (self.setpoints.yaw_set < 0):
			self.setpoints.yaw_set += 360

		self.setpoints.pitch_set = self.smach_data.pitch_set #changed to absolute
		self.setpoints.roll_set = self.smach_data.roll_set #changed to absolute
		self.setpoints.depth_set = self.smach_data.depth_set + self.sensors_data.depth_current
		self.setpoints.distance_set = self.smach_data.distance_set


	def surface(self):
		print("surfacing")
		self.setpoints.yaw_set = self.sensors_data.yaw_current
		self.setpoints.pitch_set, self.setpoints.roll_set = 0, 0
		self.setpoints.distance_set, self.setpoints.depth_set = 0, 0
		self.setpoints.final_command = True
		gnc_pub.publish(self.setpoints)
		sleep(10)
		exit()


	def bumpIntoBuoy(self):
		print("bumping")
		#Forwards
		self.setpoints.distance_set = 5
		gnc_pub.publish(self.setpoints)
		sleep(4)
		#Stop
		self.setpoints.distance_set = 0
		gnc_pub.publish(self.setpoints)
		sleep(1)
		#Backwards
		self.setpoints.distance_set = -5
                gnc_pub.publish(self.setpoints)
                sleep(4)
		#Stop
		self.setpoints.distance_set = 0
		gnc_pub.publish(self.setpoints)
		sleep(1)
		self.setpoints.final_command = True
		gnc_pub.publish(self.setpoints)


def main_loop():
	GNC = Guidance_Navigation_Control()
	sleep(1)
	while not (GNC.data_received()):
		sleep(0.1)
		print "no data"
	print "yes data"
	while not rospy.is_shutdown():
		sleep(0.5)
		GNC.main_loop()
	print("exiting")
	exit()


if __name__ == '__main__':
	rospy.init_node('GNC')
	gnc_pub = rospy.Publisher('controlCommand', controlCommand, queue_size=10)

	main_loop()

