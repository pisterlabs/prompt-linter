#!/usr/bin/env python
import rospy
import smach
import smach_ros
import time
from computer_vision.msg import target
from guidance_navigation_control.msg import sensorInfo_actuatorStatus
from guidance_navigation_control.msg import task_desiredAction
from Subscriber import Subscribe_to


class LostBuoy(smach.State):
	def __init__(self):
		print("lost")
		smach.State.__init__(self, outcomes=['Success', 'Failed'])
		self.smach_pub = rospy.Publisher('task_desiredAction', task_desiredAction, queue_size=10)
		self.cv_sub = Subscribe_to('target')
		self.sensors_sub = Subscribe_to('sensorInfo_actuatorStatus')
		self.task = task_desiredAction()
		time.sleep(0.1)

	def execute(self, userdata):
		self.task.currentState = "Lost: Buoy"
                self.smach_pub.publish(self.task)
		#maintain current position
		self.task.yaw_set, self.task.depth_set, self.task.distance_set = 0, 0, 0
		self.smach_pub.publish(self.task)
		self.cv_data = self.cv_sub.get_data()
		if not self.cv_data.buoy1:
			self.cv_data = self.cv_sub.get_data()
			#back up
			print("back up")
			self.task.distance_set = -2
			self.smach_pub.publish(self.task)
			time.sleep(2)
			self.task.distance_set = 0
			self.smach_pub.publish(self.task)
			#set search commands
			print("search")
			self.left = (-15, 0)
			self.right = (30, 0)
			self.origin = (-15, 0)
			self.up = (0, 0.1524)
			self.down = (0, -0.3048)
			self.movements = [self.left, self.right, self.origin, self.up, self.down]
			for i in range(len(self.movements)):
				self.moveOutcome = self.move(self.movements[i][0], self.movements[i][1])
				if (self.moveOutcome == "Buoy"):
					print("Buoy")
					return 'Success'
				elif (self.moveOutcome == "No Buoy"):
					print("No Buoy")
					continue
				elif (self.moveOutcome == "Not Stabilized"):
					print("Not Stabilized")
					return 'Failed'
			return 'Failed'

	def move(self, yaw, depth):
		self.cv_data = self.cv_sub.get_data()
		self.task.yaw_set = yaw
		self.task.depth_set = depth
		self.smach_pub.publish(self.task)
		print(yaw, depth)
		time.sleep(0.1)
		self.counter = 0
		while not self.cv_data.buoy1:
			self.cv_data = self.cv_sub.get_data()
			self.sensors_data = self.sensors_sub.get_data()
			time.sleep(0.01)
			self.counter += 1
			if self.sensors_data.stabilized:
				return "No Buoy"
			elif (self.counter > 2000):
				return "Not Stabilized"
		return "Buoy"

def code():
        rospy.init_node('sm')
        main = smach.StateMachine(outcomes=['Done', 'Not_Done'])
        with main:
                smach.StateMachine.add('LostBuoy', LostBuoy(), transitions={ 'Success':'Done',
										'Failed':'Not_Done'})

        sis = smach_ros.IntrospectionServer('server', main, '/tester')
        sis.start()
        outcome = main.execute()
        sis.stop()
        rospy.spin()


if __name__ == '__main__':
	code()


