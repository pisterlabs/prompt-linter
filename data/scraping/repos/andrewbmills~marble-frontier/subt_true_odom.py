#!/usr/bin/env python
import sys
import numpy as np
import guidance
import rospy
import tf
from nav_msgs.msg import *
from geometry_msgs.msg import *
from gazebo_msgs.srv import GetLinkState

class LinkStateToOdometry:
	def getLinkState(self): # Position subscriber callback function		
		X1_state = self.gazebo_link_state(self.link_name, 'world')
		self.Odometry.pose.pose = X1_state.link_state.pose
		self.Odometry.twist.twist = X1_state.link_state.twist

		# Add time stamp
		self.Odometry.header.stamp = rospy.Time.now()

		self.broadcaster.sendTransform((self.Odometry.pose.pose.position.x, self.Odometry.pose.pose.position.y, self.Odometry.pose.pose.position.z),
										(self.Odometry.pose.pose.orientation.x, self.Odometry.pose.pose.orientation.y, self.Odometry.pose.pose.orientation.z, self.Odometry.pose.pose.orientation.w),
										rospy.Time.now(), "X1/base_link", "world")

		return

	def start(self):
		rate = rospy.Rate(20.0) # 50Hz
		while not rospy.is_shutdown():
			rate.sleep()
			self.getLinkState()
			self.pub1.publish(self.Odometry)
		return

	def __init__(self, link_name="base_link", topic_name="odometry", robot_name="X1", frame="world", child_frame="X1/base_link"):
		
		node_name = topic_name+ "_" + robot_name
		rospy.init_node(node_name)

		self.link_name = robot_name + "::" + robot_name + "/" + link_name
		self.pubTopic1 = "/" + robot_name + "/" + topic_name
		self.pub1 = rospy.Publisher(self.pubTopic1, Odometry, queue_size=10)

		# Initialize Odometry message object
		self.Odometry = Odometry()
		self.Odometry.header.seq = 1
		self.Odometry.header.frame_id = frame
		self.Odometry.child_frame_id = child_frame

		# Initialize Gazebo LinkState service
		rospy.wait_for_service('/gazebo/get_link_state')
		self.gazebo_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState, persistent=True)

		# TF Broadcaster
		self.broadcaster = tf.TransformBroadcaster()

if __name__ == '__main__':
	publish_tool = LinkStateToOdometry()

	try:
		publish_tool.start()
	except rospy.ROSInterruptException:
		pass