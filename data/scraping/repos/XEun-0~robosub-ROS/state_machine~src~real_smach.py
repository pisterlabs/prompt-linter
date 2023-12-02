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

from Initialize_Robot import Initialize_Robot
from SearchForBuoy import SearchForBuoy
from CenterWithBuoy import CenterWithBuoy
from LostBuoy import LostBuoy
from BumpIntoBuoy import BumpIntoBuoy
from SearchForBase import SearchForBase
from Surface import Surface


def the_loop():
	print "State Machine Starting"
	time.sleep(1)
	my_machine = smach.StateMachine(outcomes=['StopRobot'])
	with my_machine:
		smach.StateMachine.add('Initialize_Robot', Initialize_Robot(), transitions={'Success':'SearchForBuoy', 'Failed':'Surface'})
		smach.StateMachine.add('SearchForBuoy', SearchForBuoy(), transitions={'Success':'CenterWithBuoy', 'Failed':'Surface'})
		smach.StateMachine.add('CenterWithBuoy', CenterWithBuoy(), transitions={'Success':'BumpIntoBuoy', 'Lost':'LostBuoy', 'Failed':'Surface'})
		smach.StateMachine.add('LostBuoy', LostBuoy(), transitions={'Success':'CenterWithBuoy', 'Failed':'Surface'})
		smach.StateMachine.add('BumpIntoBuoy', BumpIntoBuoy(), transitions={'Success':'SearchForBase', 'Failed':'Surface'})
		smach.StateMachine.add('SearchForBase', SearchForBase(), transitions={'Success':'Surface', 'Failed':'Surface'})
		smach.StateMachine.add('Surface', Surface(), transitions={'Success':'StopRobot'})

	sis = smach_ros.IntrospectionServer('server', my_machine, '/Arens_State_Machine')
	sis.start()
	outcome = my_machine.execute()
	rospy.spin()
	sis.stop()


if __name__ == '__main__':
	rospy.init_node('SMACH')
	the_loop()
