#!/usr/bin/env python
import rospy
import smach
import smach_ros
import time
from guidance_navigation_control.msg import task_desiredAction
from Subscriber import Subscribe_to


class Surface(smach.State):
	def __init__(self):
		print("surfacing")
		smach.State.__init__(self, outcomes=['Success'])
		self.smach_pub = rospy.Publisher('task_desiredAction', task_desiredAction, queue_size=10)
		self.task = task_desiredAction()
		time.sleep(0.1)

	def execute(self, userdata):
		self.task.currentState = "Surface"
                self.smach_pub.publish(self.task)
		#Need bool since depth adds to current not absolute
		self.task.surface = True
		self.smach_pub.publish(self.task)
		time.sleep(5)
		return 'Success'
		#Is there a way to know we surfaced for sure


def code():
        rospy.init_node('sm')
        main = smach.StateMachine(outcomes=['Done', 'Not_Done'])
        with main:
                smach.StateMachine.add('Surface', Surface(), transitions={'Success':'Done'})

        sis = smach_ros.IntrospectionServer('server', main, '/tester')
        sis.start()
        outcome = main.execute()
        sis.stop()
        rospy.spin()


if __name__ == '__main__':
	code()

