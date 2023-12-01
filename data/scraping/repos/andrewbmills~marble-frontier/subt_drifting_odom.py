#!/usr/bin/env python
import sys
import numpy as np
import guidance
import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion
from gazebo_msgs.srv import GetLinkState
from copy import deepcopy

def quaternion2euler(q):
	euler = np.array([0.0, 0.0, 0.0])
	euler[0] = np.arctan2(2*(q.w*q.x + q.y*q.z), 1.0 - 2.0*(q.x*q.x + q.y*q.y))
	euler[1] = np.arcsin(2*(q.w*q.y - q.z*q.x))
	euler[2] = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
	return euler

def quaternion2rotationmatrix(q):
	R = np.zeros((3,3))
	R[0,0] = q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z
	R[0,1] = 2*(q.x*q.y - q.w*q.z)
	R[0,2] = 2*(q.w*q.y + q.x*q.z)
	R[1,0] = 2*(q.x*q.y + q.w*q.z)
	R[1,1] = q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z
	R[1,2] = 2*(q.y*q.z - q.w*q.x)
	R[2,0] = 2*(q.x*q.z - q.w*q.y)
	R[2,1] = 2*(q.w*q.x + q.y*q.z)
	R[2,2] = q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z
	return R

def euler2quaternion(euler):
	q = Quaternion()
	cy = np.cos(euler[2]*0.5)
	sy = np.sin(euler[2]*0.5)
	cp = np.cos(euler[1]*0.5)
	sp = np.sin(euler[1]*0.5)
	cr = np.cos(euler[0]*0.5)
	sr = np.sin(euler[0]*0.5)

	q.w = cy*cp*cr + sy*sp*sr
	q.x = cy*cp*sr - sy*sp*cr
	q.y = sy*cp*sr + cy*sp*cr
	q.z = sy*cp*cr - cy*sp*sr
	return q

def euler2rotationmatrix(euler):
	R = np.zeros((3,3))
	r = euler[0] # roll
	p = euler[1] # pitch
	y = euler[2] # yaw
	R[0,0] = np.cos(p)*np.cos(y)
	R[0,1] = -np.cos(r)*np.sin(y) + np.sin(r)*np.sin(p)*np.cos(y)
	R[0,2] = np.sin(r)*np.sin(y) + np.cos(r)*np.sin(p)*np.cos(y)
	R[1,0] = np.cos(p)*np.sin(y)
	R[1,1] = np.cos(r)*np.cos(y) + np.sin(r)*np.sin(p)*np.sin(y)
	R[1,2] = -np.sin(r)*np.cos(y) + np.cos(r)*np.sin(p)*np.sin(y)
	R[2,0] = -np.sin(p)
	R[2,1] = np.sin(r)*np.cos(p)
	R[2,2] = np.cos(r)*np.cos(p)
	return R

def angleDiff(a, b):
	# Computes a-b, preserving the correct sign (counter-clockwise positive angles)
	# All angles are in degrees
	a = (360000 + a) % 360
	b = (360000 + b) % 360
	d = a - b
	d = (d + 180) % 360 - 180
	return d

class LinkStateToOdometry:
	def getLinkState(self): # Position subscriber callback function		
		X1_state = self.gazebo_link_state(self.link_name, 'world')
		self.Odometry.pose.pose = X1_state.link_state.pose
		self.Odometry.twist.twist = X1_state.link_state.twist

		# Add time stamp
		self.Odometry.header.stamp = rospy.Time.now()

		return

	def sendTransform(self):
		self.br.sendTransform((self.Odometry_Noisy.pose.pose.position.x,
					self.Odometry_Noisy.pose.pose.position.y,
					self.Odometry_Noisy.pose.pose.position.z),
				(self.Odometry_Noisy.pose.pose.orientation.x,
					self.Odometry_Noisy.pose.pose.orientation.y,
					self.Odometry_Noisy.pose.pose.orientation.z,
					self.Odometry_Noisy.pose.pose.orientation.w),
				rospy.Time.now(),
				self.robot_tf_name,
				self.Odometry_Noisy.header.frame_id)
		return
	def addNoise(self):
		# You should probably redo the way you're doing this by just simulating an integrated velocity with a small but nonzero xdot, ydot, zdot, yawdot) mean
		# and then sampling a normal distribution in position and orientation about it.

		# Set Odometry_Noisy to the truth
		self.Odometry_Noisy = deepcopy(self.Odometry)

		# Get euler angles from true odometry
		euler = quaternion2euler(self.Odometry.pose.pose.orientation)

		# Migrate the bias based on a constant rate of migration and the time since last update
		# Time since last update
		dt = 1/self.rate
		# dt = self.Odometry.header.stamp - self.lastOdometry.header.stamp
		# dt = dt.to_sec()

		# Sample the normal walk distribution for the bias migration
		sigma_bias_migrate = self.Odometry_migrate_rate_sigma*dt
		bias_migrate_rate = np.random.normal(np.zeros(6), self.Odometry_migrate_rate_sigma, 6)
		self.Odometry_bias = self.Odometry_bias + bias_migrate_rate*dt

		# Sample the normal distribution around the bias
		noise = np.random.normal(np.zeros(6), self.sigma_noise, 6)
		euler_noisy = euler + noise[3:6]
		q_current_pose_noisy = euler2quaternion(euler_noisy)

		self.Odometry_Noisy.pose.pose.position.x = self.Odometry_Noisy.pose.pose.position.x + noise[0] + self.Odometry_bias[0]
		self.Odometry_Noisy.pose.pose.position.y = self.Odometry_Noisy.pose.pose.position.y + noise[1] + self.Odometry_bias[1]
		self.Odometry_Noisy.pose.pose.position.z = self.Odometry_Noisy.pose.pose.position.z + noise[2] + self.Odometry_bias[2]
		self.Odometry_Noisy.pose.pose.orientation.x = q_current_pose_noisy.x
		self.Odometry_Noisy.pose.pose.orientation.y = q_current_pose_noisy.y
		self.Odometry_Noisy.pose.pose.orientation.z = q_current_pose_noisy.z
		self.Odometry_Noisy.pose.pose.orientation.w = q_current_pose_noisy.w

		# Set lastOdometry to current for next loop
		self.lastOdometry = deepcopy(self.Odometry)

		if (self.loops % 100) == 0:
			print("Odometry_bias = [%0.3f m, %0.3f m, %0.3f m, %0.3f deg, %0.3f deg, %0.3f deg]" % (self.Odometry_bias[0], self.Odometry_bias[1], self.Odometry_bias[2], self.Odometry_bias[3]*(180.0/np.pi), self.Odometry_bias[4]*(180.0/np.pi), self.Odometry_bias[5]*(180.0/np.pi)))
			# print("True Odom = [x = %0.2f m, y = %0.2f m, z = %0.2f m,/nq.x = %0.2f, q.y= %0.2f, q.z = %0.2f, q.w = %0.2f]" % (self.Odometry.pose.pose.position.x, self.Odometry.pose.pose.position.y, self.Odometry.pose.pose.position.z, self.Odometry.pose.pose.orientation.x, self.Odometry.pose.pose.orientation.y, self.Odometry.pose.pose.orientation.z, self.Odometry.pose.pose.orientation.w))
			# print("Noisy Odom = [x = %0.2f m, y = %0.2f m, z = %0.2f m,/nq.x = %0.2f, q.y= %0.2f, q.z = %0.2f, q.w = %0.2f]" % (self.Odometry_Noisy.pose.pose.position.x, self.Odometry_Noisy.pose.pose.position.y, self.Odometry_Noisy.pose.pose.position.z, self.Odometry_Noisy.pose.pose.orientation.x, self.Odometry_Noisy.pose.pose.orientation.y, self.Odometry_Noisy.pose.pose.orientation.z, self.Odometry_Noisy.pose.pose.orientation.w))

		self.loops = self.loops + 1

	def start(self):
		rate = rospy.Rate(self.rate) # 50Hz
		while not rospy.is_shutdown():
			rate.sleep()
			self.getLinkState()
			self.addNoise()
			self.sendTransform()
			self.pub1.publish(self.Odometry_Noisy)
		return

	def __init__(self, link_name="base_link", topic_name="odometry_noisy", robot_name="X1", frame="world", child_frame="X1/base_link", sigma_noise = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
		
		node_name = topic_name+ "_" + robot_name
		rospy.init_node(node_name)

		self.robot_tf_name = robot_name + "/" + link_name
		self.link_name = robot_name + "::" + robot_name + "/" + link_name
		self.pubTopic1 = "/" + robot_name + "/" + topic_name
		self.pub1 = rospy.Publisher(self.pubTopic1, Odometry, queue_size=10)

		# Initialize rate
		self.rate = 20.0 # Hz

		# Initialize Gazebo LinkState service
		rospy.wait_for_service('/gazebo/get_link_state')
		self.gazebo_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState, persistent=True)

		# Initialize Odometry message object
		self.Odometry = Odometry()
		self.Odometry.header.seq = 1
		self.Odometry.header.frame_id = frame
		self.Odometry.child_frame_id = child_frame

		# Initialize Odometry bias term and a sigma for the Gaussian from which the migration rate is sampled
		self.Odometry_bias = np.zeros(6) # [x, y, z, roll, pitch, yaw]: (m, m, m, rad, rad, rad)
		self.Odometry_migrate_rate_sigma = np.zeros(6)
		self.Odometry_migrate_rate_sigma[0] = 30.0/60.0 # m/s
		self.Odometry_migrate_rate_sigma[1] = 30.0/60.0 # m/s
		self.Odometry_migrate_rate_sigma[2] = 0.0/60.0 # m/s
		self.Odometry_migrate_rate_sigma[5] = (np.pi/180.0)*20.0/60.0 # rad/s

		# Noisy Odometry message object
		self.Odometry_Noisy = Odometry()
		self.Odometry_Noisy.header.seq = 1
		self.Odometry_Noisy.header.frame_id = frame
		self.Odometry_Noisy.child_frame_id = child_frame

		# Gaussian noise parameters for general noise around the current truth + bias
		self.sigma_noise = sigma_noise

		# Print Screen loop counter
		self.loops = 0

		# TF transformer
		self.br = tf.TransformBroadcaster()

if __name__ == '__main__':
	publish_tool = LinkStateToOdometry()

	try:
		publish_tool.start()
	except rospy.ROSInterruptException:
		pass