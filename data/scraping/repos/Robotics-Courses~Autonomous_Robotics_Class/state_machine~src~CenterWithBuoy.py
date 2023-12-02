#!/usr/bin/env python
import rospy
import smach
import smach_ros
import time
from computer_vision.msg import target
from guidance_navigation_control.msg import sensorInfo_actuatorStatus
from guidance_navigation_control.msg import task_desiredAction
from Subscriber import Subscribe_to


class CenterWithBuoy(smach.State):
	def __init__(self):
		print("centering")
		smach.State.__init__(self, outcomes=['Success', 'Lost', 'Failed'])
		self.smach_pub = rospy.Publisher('task_desiredAction', task_desiredAction, queue_size=10)
		self.cv_sub = Subscribe_to('target')
		self.sensors_sub = Subscribe_to('sensorInfo_actuatorStatus')
		self.task = task_desiredAction()
		self.centered = False
		self.counter = 0
		time.sleep(0.1)

	def execute(self, userdata):
		self.task.currentState = "Center: Buoy"
                self.smach_pub.publish(self.task)
		self.cv_data = self.cv_sub.get_data()
		while self.cv_data.buoy1:
			self.cv_data = self.cv_sub.get_data()
			self.sensors_data = self.sensors_sub.get_data()
			self.center()
			# X less than 5 degrees and Y less than 6 inches (Test These Numbers)
			if ((abs(self.cv_data.buoy1x) < 5) and (abs(self.cv_data.buoy1y) < 0.1524)):
				self.task.yaw_set = 0
				self.task.depth_set = 0
				self.smach_pub.publish(self.task)
				if (self.sensors_data.stabilized):
					self.centered = True
			else:
				self.centered = False

			if self.centered:
				#Center 1 meter from the Buoy
				self.task.distance_set = self.cv_data.buoy1_distance - 1
				self.smach_pub.publish(self.task)
				if (abs(self.cv_data.buoy1_distance) < 0.4):  #changed 0.4 <- .1524
					self.task.distance_set = 0
					self.smach_pub.publish(self.task)
					if (self.sensors_data.stabilized):
						return 'Success'
			self.counter = self.counter + 1
			time.sleep(0.01)
			if (self.counter > 6000):
				return 'Failed'
		return 'Lost'

	def center(self):
		self.task.yaw_set = self.cv_data.buoy1x
		self.task.depth_set = self.cv_data.buoy1y
		#In case we become uncentered
		if not (self.centered):
			self.task.distance_set = 0
		self.smach_pub.publish(self.task)


def code():
        rospy.init_node('sm')
        main = smach.StateMachine(outcomes=['Done', 'Not_Done', 'Sorta_Done'])
        with main:
                smach.StateMachine.add('CenterWithBuoy', CenterWithBuoy(), transitions={ 'Success':'Done',
							'Lost':'Sorta_Done', 'Failed':'Not_Done'})

        sis = smach_ros.IntrospectionServer('server', main, '/tester')
        sis.start()
        outcome = main.execute()
        sis.stop()
        rospy.spin()


if __name__ == '__main__':
	code()


