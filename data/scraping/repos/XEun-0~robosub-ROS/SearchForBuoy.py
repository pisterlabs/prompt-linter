#!/usr/bin/env python
import rospy
import smach
import smach_ros
import time
from computer_vision.msg import target
from guidance_navigation_control.msg import task_desiredAction
from guidance_navigation_control.msg import sensorInfo_actuatorStatus
from Subscriber import Subscribe_to


class SearchForBuoy(smach.State):
	def __init__(self):
		print("searching")
		smach.State.__init__(self, outcomes=['Success', 'Failed'])
		self.smach_pub = rospy.Publisher('task_desiredAction', task_desiredAction, queue_size=10)
		self.cv_sub = Subscribe_to('target')
		self.sensors_sub = Subscribe_to('sensorInfo_actuatorStatus')
		self.counter = 0
		self.gate_detected = False
		self.task = task_desiredAction()
		time.sleep(2)


	def execute(self, userdata):
		self.task.currentState = "Search: Buoy"
		print("Dive")
		self.task.depth_set = 1
		self.smach_pub.publish(self.task)
		self.task.depth_set = 0
		time.sleep(3)

		self.sensors_data = self.sensors_sub.get_data()
		while self.sensors_data.stabilized_time == 0:
			time.sleep(0.01)
			self.counter = self.counter + 1
			if (self.counter > 3000): #was originally 15
				return 'Failed'
			self.sensors_data = self.sensors_sub.get_data()

		self.cv_data = self.cv_sub.get_data()
		for target in self.cv_data.targets:
			if target.name == 'Gate_GMan':
				self.gate_detected = True
		if not self.gate_detected:
			rotation = 720
			angle_increment = 30
			counter = 0
			rate = rospy.Rate(20)
			while counter <= (rotation / angle_increment) and not self.gate_detected:
				self.sensors_data = self.sensors_sub.get_data()
				if self.sensors_data.stabilized_time > 0:
					self.task.yaw_set = angle_increment
					print (self.task.yaw_set + self.sensors_data.yaw_current)
					self.smach_pub.publish(self.task)
					#time.sleep(0.2) might be needed
					counter += 1

				self.cv_data = self.cv_sub.get_data()
				for target in self.cv_data.targets:
					if target.name == 'Gate_GMan':
						self.gate_detected = True
				rate.sleep()

		if self.gate_detected:
			return 'Success'

		return 'Failed'


def code():
        rospy.init_node('sm')
        main = smach.StateMachine(outcomes=['Done', 'Not_Done'])
        with main:
                smach.StateMachine.add('SearchForBuoy', SearchForBuoy(), transitions={ 'Success':'Done',
										'Failed':'Not_Done'})

        sis = smach_ros.IntrospectionServer('server', main, '/tester')
        sis.start()
        outcome = main.execute()
        sis.stop()
        rospy.spin()


if __name__ == '__main__':
	code()
