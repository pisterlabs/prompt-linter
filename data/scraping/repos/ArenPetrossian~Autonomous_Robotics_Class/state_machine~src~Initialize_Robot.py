#!/usr/bin/env python
import rospy
import smach
import smach_ros
import time
from computer_vision.msg import target
from guidance_navigation_control.msg import sensorInfo_actuatorStatus
from guidance_navigation_control.msg import task_desiredAction
from guidance_navigation_control.msg import controlCommand
from Subscriber import Subscribe_to


class Initialize_Robot(smach.State):
	def __init__(self):
		print("starting")
		smach.State.__init__(self, outcomes=['Success', 'Failed'])
		#Subscribe to all nodes that publish to smach
		self.cv_sub = Subscribe_to('target')
		self.sensors_sub = Subscribe_to('sensorInfo_actuatorStatus')
		self.gnc_sub = Subscribe_to('controlCommand')
		self.counter = 0
		time.sleep(2)

	def execute(self, userdata):
		#Check if all nodes have published data
		print (self.cv_sub.was_data_sent(), self.sensors_sub.was_data_sent(), self.gnc_sub.was_data_sent())
		#If any of the nodes are not publishing, stay in this loop
		while not (self.cv_sub.was_data_sent() and self.sensors_sub.was_data_sent() and self.gnc_sub.was_data_sent()):
			time.sleep(0.01)
			#Continue checking if all nodes have published data
			print (self.cv_sub.was_data_sent(), self.sensors_sub.was_data_sent(), self.gnc_sub.was_data_sent())
			#If any nodes have failed to publish after ~20 seconds, return failed
			if (self.counter > 2000):
				return 'Failed'
			self.counter = self.counter + 1
		#When all nodes are publishing data, return Finished
		return 'Success'


def code():
        rospy.init_node('sm')
        main = smach.StateMachine(outcomes=['Done', 'Not_Done'])
        with main:
                smach.StateMachine.add('Initialize_Robot', Initialize_Robot(), transitions={ 'Success':'Done',
										'Failed':'Not_Done'})

        sis = smach_ros.IntrospectionServer('server', main, '/tester')
        sis.start()
        outcome = main.execute()
        sis.stop()
        rospy.spin()


if __name__ == '__main__':
	code()


