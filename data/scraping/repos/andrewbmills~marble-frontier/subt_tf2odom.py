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
	def getTransform(self): # Position subscriber callback function
		(trans,rot) = self.tf_listener.lookupTransform(self.Odometry.header.frame_id, self.Odometry.child_frame_id, rospy.Time(0))
		self.Odometry.pose.pose.position.x = trans[0]
		self.Odometry.pose.pose.position.y = trans[1]
		self.Odometry.pose.pose.position.z = trans[2]
		self.Odometry.pose.pose.orientation.x = rot[0]
		self.Odometry.pose.pose.orientation.y = rot[1]
		self.Odometry.pose.pose.orientation.z = rot[2]
		self.Odometry.pose.pose.orientation.w = rot[3]

		# Add time stamp
		self.Odometry.header.stamp = rospy.Time.now()

		self.Odometry.pose.pose.position.z = self.Odometry.pose.pose.position.z

		# Assign pose topic values
		self.Pose.pose.pose.position = self.Odometry.pose.pose.position
		self.Pose.pose.pose.orientation = self.Odometry.pose.pose.orientation
		return

	def start(self):
		rate = rospy.Rate(25.0) # 50Hz
		while not rospy.is_shutdown():
			rate.sleep()
			self.getTransform()
			self.pub1.publish(self.Odometry)
			self.pub2.publish(self.Pose)
		return

	def __init__(self, topic_name="odometry", robot_name="X4", fixed_frame="world", child_frame="X4/base_link"):
		
		node_name = topic_name + "_" + robot_name + "_publisher"
		rospy.init_node(node_name)

		self.pubTopic1 = "/" + robot_name + "/" + topic_name
		self.pub1 = rospy.Publisher(self.pubTopic1, Odometry, queue_size=10)
		self.pubTopic2 = "/" + robot_name + "/pose"
		self.pub2 = rospy.Publisher(self.pubTopic2, PoseWithCovarianceStamped, queue_size=10)

		# Initialize Odometry message object
		self.Odometry = Odometry()
		self.Odometry.header.seq = 1
		self.Odometry.header.frame_id = fixed_frame
		self.Odometry.child_frame_id = child_frame

		# Initialize Pose message object
		self.Pose = PoseWithCovarianceStamped()
		self.Pose.header.seq = 1
		self.Pose.header.frame_id = fixed_frame
		covariance = [0.0]*36
		covariance[0] = 0.001 # meters
		covariance[7] = 0.001 # meters
		covariance[14] = 0.001 # meters
		covariance[21] = 0.0175 # rad
		covariance[28] = 0.0175 # rad
		covariance[35] = 0.0175 # rad
		self.Pose.pose.covariance = covariance

		# Initialize tf listener
		self.tf_listener = tf.TransformListener()


if __name__ == '__main__':
	publish_tool = LinkStateToOdometry()

	try:
		publish_tool.start()
	except rospy.ROSInterruptException:
		pass