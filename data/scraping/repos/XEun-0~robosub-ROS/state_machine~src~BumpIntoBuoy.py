#!/usr/bin/env python
import rospy
import smach
import smach_ros
import time
from guidance_navigation_control.msg import sensorInfo_actuatorStatus
from guidance_navigation_control.msg import task_desiredAction
from guidance_navigation_control.msg import controlCommand
from Subscriber import Subscribe_to


class BumpIntoBuoy(smach.State):
	def __init__(self):
		print("bumping")
		smach.State.__init__(self, outcomes=['Success', 'Failed'])
		self.smach_pub = rospy.Publisher('task_desiredAction', task_desiredAction, queue_size=10)
		self.sensors_sub = Subscribe_to('sensorInfo_actuatorStatus')
		self.gnc_sub = Subscribe_to('controlCommand')
		self.task = task_desiredAction()
		self.counter = 0
		time.sleep(0.1)

	def execute(self, userdata):
		self.task.currentState = "Bump: Buoy"
                self.smach_pub.publish(self.task)
		self.task.bumpIntoBuoy = True
		self.smach_pub.publish(self.task)
		while True:
			self.sensors_data = self.sensors_sub.get_data()
			self.gnc_data = self.gnc_sub.get_data()
			if (self.sensors_data.stabilized and self.gnc_data.final_command):
				return 'Success'
			#Change this according to the GNC's bumpIntoBuoy function
			self.counter = self.counter + 1
			time.sleep(0.01)
			if (self.counter > 2000):
				return 'Failed'


def code():
        rospy.init_node('sm')
        main = smach.StateMachine(outcomes=['Done', 'Not_Done'])
        with main:
                smach.StateMachine.add('BumpIntoBuoy', BumpIntoBuoy(), transitions={ 'Success':'Done',
										'Failed':'Not_Done'})

        sis = smach_ros.IntrospectionServer('server', main, '/tester')
        sis.start()
        outcome = main.execute()
        sis.stop()
        rospy.spin()


if __name__ == '__main__':
	code()


