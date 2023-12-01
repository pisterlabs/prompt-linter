#!/usr/bin/env python
import sys
import numpy as np
import guidance
import rospy
import math
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

class guidance_controller:
	def getPosition(self, data):
		self.position = data.pose.pose.position
		q = Quaternion()
		q = data.pose.pose.orientation
		self.yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
		self.R = np.zeros((3,3))
		self.R[0,0] = q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z
		self.R[0,1] = 2.0*(q.x*q.y - q.w*q.z)
		self.R[0,2] = 2.0*(q.w*q.y + q.x*q.z)

		self.R[1,0] = 2.0*(q.x*q.y + q.w*q.z)
		self.R[1,1] = q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z
		self.R[1,2] = 2.0*(q.y*q.z - q.w*q.x)

		self.R[2,0] = 2.0*(q.x*q.z - q.w*q.y)
		self.R[2,1] = 2.0*(q.w*q.x + q.y*q.z)
		self.R[2,2] = q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z

		self.positionUpdated = 1
		return

	def getPath(self, data): # Path subscriber callback function
		newpath = np.empty((3,len(data.poses)))
		for i in range(0,len(data.poses)):
			newpath[0,i] = data.poses[i].pose.position.x
			newpath[1,i] = data.poses[i].pose.position.y
			newpath[2,i] = data.poses[i].pose.position.z
			if (math.isnan(newpath[0,i]) or math.isnan(newpath[1,i]) or math.isnan(newpath[2,i])):
				print("point [%0.1f, %0.1f, %0.1f], entry %d in path of length %d, contained NaNs" % (newpath[0,i], newpath[1,i], newpath[2,i], i, len(data.poses)))
				return
		self.path = newpath
		print('path received of size: %d' % self.path.shape[1])
		self.pathUpdated = 1
		return

	def getGoalPose(self, data): # Goal Pose subscriber callback function
		q = Quaternion()
		q = data.pose.orientation
		self.goal_yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
		# print("Goal pose yaw is %0.2f" % ((180.0/np.pi)*self.goal_yaw))
		return

	def publishLookahead(self):
		# Remove the old marker
		self.L2_marker.action = 2
		self.pub2.publish(self.L2_marker)

		# Add a new one
		self.L2_marker.action = 0
		self.L2_marker.points[0] = self.position
		self.L2_marker.points[1] = self.L2
		self.pub2.publish(self.L2_marker)
		return

	def updateCommand(self): # Updates the twist command for publishing
		# Check if the subscribers have updated the robot position and path
		if (self.path.shape[1] < 1):
			print("No guidance command, path is empty.")
			return

		if (self.positionUpdated == 0):
			print("No odometry message.")
			return
		# Convert the body frame x-axis of the robot to inertial frame
		heading_body = np.array([[1.0], [0.0], [0.0]]) # using commanded velocity for now (use actual later)
		heading_inertial = np.matmul(self.R, heading_body)
		velocity_inertial = self.speed*np.array([heading_inertial[0,0], heading_inertial[1,0], heading_inertial[2,0]])

		# Find the lookahead/carrot point for the guidance controller
		# Store the vehicle position for now
		p_robot = np.array([self.position.x, self.position.y, self.position.z])
		path = self.path
		start = np.array([path[0,0], path[1,0], path[2,0]])
		goal = np.array([path[0,-1], path[1,-1], path[2,-1]])
		if (self.path.shape[1] < 2):
			p_L2 = goal
			v_L2 = (goal - p_robot)/np.linalg.norm(goal - p_robot)
			L2_vec = p_L2[0:2] - p_robot[0:2]
			print("Path is only one point long, heading to goal point.")
		else:
			if (self.vehicle_type == 'ground'):
				p_L2, v_L2 = guidance.find_Lookahead_Discrete_2D(path[0:2,:], p_robot[0:2], self.speed*self.Tstar, 0, 0, reverse=self.reverse)

				# If p_L2 is the start of the path, check if the goal point is within an L2 radius of the vehicle, if so, go to the goal point
				if (np.linalg.norm(p_robot[0:2] - goal[0:2]) <= 1.02*self.speed*self.Tstar):
					p_L2 = goal

				# print("The L2 point is: [%0.2f, %0.2f]" % (p_L2[0], p_L2[1]))
				# Update class members
				self.L2.x = p_L2[0]
				self.L2.y = p_L2[1]
				self.L2.z = p_robot[2]
				L2_vec = p_L2[0:2] - p_robot[0:2]
			else:
				p_L2, v_L2 = guidance.find_Lookahead_Discrete_3D(path, p_robot, self.speed*self.Tstar, 0, 0, reverse=self.reverse)
				# If p_L2 is the start of the path, check if the goal point is within an L2 radius of the vehicle, if so, go to the goal point
				if (np.linalg.norm(p_robot - goal) <= 1.02*self.speed*self.Tstar):
					p_L2 = goal

				# Update class members
				self.L2.x = p_L2[0]
				self.L2.y = p_L2[1]
				self.L2.z = p_L2[2]
				L2_vec = p_L2[0:3] - p_robot[0:3]

			# Edit later to use proportional control to just command to the goal point and the goal pose!

		# Generate a lateral acceleration command from the lookahead point
		if (self.controller_type == 'trajectory_shaping'):
			a_cmd = guidance.trajectory_Shaping_Guidance(np.array([p_L2[0], p_L2[1]]), p_robot[0:2], \
													 np.array([velocity_inertial[0], velocity_inertial[1]]), np.array([v_L2[0], v_L2[1]]))
			chi_dot = -a_cmd/self.speed
			if (self.vehicle_type == 'air'):
				chi_dot = -chi_dot # reverse convention
		else:
			if (self.vehicle_type == 'ground'):
				a_cmd = guidance.L2_Plus_Guidance_2D(np.array([p_L2[0], p_L2[1]]), p_robot[0:2], \
													 np.array([velocity_inertial[0], velocity_inertial[1]]), self.Tstar, 0)
				chi_dot = -a_cmd/self.speed
			else:
				# a_cmd = guidance.L2_Plus_Guidance_3D(p_L2, p_robot, velocity_inertial, self.Tstar, 0)
				# Convert lateral acceleration to angular acceleration about the z axis
				# chi_dot = a_cmd[1]/self.speed
				a_cmd = guidance.L2_Plus_Guidance_2D(np.array([p_L2[0], p_L2[1]]), p_robot[0:2], \
													 np.array([velocity_inertial[0], velocity_inertial[1]]), self.Tstar, 0)
				chi_dot = -a_cmd/self.speed

		# Change what the vehicle does depending on the path orientation relative to the robot
		dot_prod = np.dot(L2_vec[0:2], heading_inertial[0:2])/(np.linalg.norm(L2_vec[0:2])*np.linalg.norm(heading_inertial[0:2]))
		# print("The heading vector in 2D is: [%0.2f, %0.2f]" % (heading_inertial[0], heading_inertial[1]))
		# print("The robot position is : [%0.2f, %0.2f]" % (p_robot[0], p_robot[1]))
		# print("The L2 vector in 2D is: [%0.2f, %0.2f]" % (L2_vec[0], L2_vec[1]))
		# print("The goal pose heading vector in 2D is: [%0.2f, %0.2f]" % (np.cos(self.goal_yaw), np.sin(self.goal_yaw)))
		# print("cos(eta) = %0.2f" % dot_prod)
		if (dot_prod > (.5)):
			self.command.linear.x = self.speed
			self.command.angular.z = chi_dot
		# elif (dot_prod < -0.5):
		# 	self.command.linear.x = -self.speed
		# 	self.command.angular.z = chi_dot
		else:
			self.command.linear.x = 0.0
			self.command.angular.z = chi_dot

		# Do altitude control for air vehicles
		if self.vehicle_type == 'air':
			error = L2_vec[2]
			self.command.linear.z = self.gain_z*error

		if (self.vehicle_type == 'air'):
			if (np.linalg.norm(p_L2 - goal) <= 0.6):
				# Use proportional control to control to goal point
				error = L2_vec[0]*np.cos(self.yaw) + L2_vec[1]*np.sin(self.yaw)
				self.command.linear.x = self.gain_z*error
				error = L2_vec[1]*np.cos(self.yaw) - L2_vec[0]*np.sin(self.yaw)
				self.command.linear.y = self.gain_z*error
				error = (np.pi/180.0)*guidance.angle_Diff((180.0/np.pi)*self.goal_yaw, (180.0/np.pi)*self.yaw)
				self.command.angular.z = self.gain_yaw*error
		elif (np.linalg.norm(L2_vec) <= 0.3*self.Tstar*self.speed):
			error = (np.pi/180.0)*guidance.angle_Diff((180.0/np.pi)*self.goal_yaw, (180.0/np.pi)*self.yaw)
			self.command.angular.z = self.gain_yaw*error
			# print("Yaw error of %0.2f deg." % ((180.0/np.pi)*error))
			self.command.linear.x = 0.0;

		# Set Lookahead point
		self.lookahead_point.header.stamp = rospy.Time.now()
		self.lookahead_point.point = self.L2

		return

	def __init__(self):
		# Booleans for first subscription receive
		self.positionUpdated = 0
		self.pathUpdated = 0

		# Initialize ROS node and Subscribers
		node_name = 'guidance_controller'
		rospy.init_node(node_name)

		# Params
		self.vehicle_type = rospy.get_param('guidance_controller/vehicle_type', 'ground')
		self.controller_type = rospy.get_param('guidance_controller/controller_type', 'L2')
		self.fixed_frame = rospy.get_param('guidance_controller/fixed_frame', 'world')
		self.speed = rospy.get_param('guidance_controller/speed', 1.0) # m/s
		self.Tstar = rospy.get_param('guidance_controller/Tstar', 1.0) # s
		self.reverse = rospy.get_param('guidance_controller/reverse', 0)
		print("Reverse = %d" % self.reverse)

		# Subscribers
		rospy.Subscriber('odometry', Odometry, self.getPosition)
		self.path = np.empty((3,0))
		rospy.Subscriber('path', Path, self.getPath)
		rospy.Subscriber('goal_pose', PoseStamped, self.getGoalPose)
		self.goal_yaw = 0.0
		self.R = np.zeros((3,3))

		# Initialize Publisher topics
		self.pubTopic1 = 'cmd_vel'
		self.pub1 = rospy.Publisher(self.pubTopic1, Twist, queue_size=10)
		self.pubTopic2 = 'lookahead_vec'
		self.pub2 = rospy.Publisher(self.pubTopic2, Marker, queue_size=10)
		self.pubTopic3 = 'lookahead_point'
		self.pub3 = rospy.Publisher(self.pubTopic3, PointStamped, queue_size=10)
		self.lookahead_point = PointStamped()
		self.lookahead_point.header.frame_id = self.fixed_frame

		# Initialize twist object for publishing
		self.command = Twist()
		self.command.linear.x = 0.0
		self.command.linear.y = 0.0
		self.command.linear.z = 0.0
		self.command.angular.x = 0.0
		self.command.angular.y = 0.0
		self.command.angular.z = 0.0

		# Initialize Lookahead vector for publishing
		self.L2 = Point()
		self.L2.x = 0.0
		self.L2.y = 0.0
		self.L2.z = 0.0
		self.position = Point()
		self.position.x = 0.0
		self.position.y = 0.0
		self.position.z = 0.0
		self.L2_marker = Marker()
		self.L2_marker.type = 4
		self.L2_marker.header.frame_id = self.fixed_frame
		self.L2_marker.header.stamp = rospy.Time()
		self.L2_marker.id = 101;
		self.L2_marker.scale.x = 0.05
		self.L2_marker.color.b = 1.0
		self.L2_marker.color.a = 1.0
		self.L2_marker.pose.orientation.w = 1.0
		self.L2_marker.action = 0
		self.L2_marker.points.append(self.position)
		self.L2_marker.points.append(self.L2)

		# Proportional Controller
		self.gain_x = 0.3
		self.gain_y = 0.3
		self.gain_yaw = 0.2

		# Altitude controller
		self.gain_z = 0.5

		# Saturation for yaw rate
		self.yaw_rate_max = 0.3

	def start(self):
		rate = rospy.Rate(10.0) # 10Hz
		while not rospy.is_shutdown():
			rate.sleep()
			self.updateCommand()
			if (np.abs(self.command.angular.z) > self.yaw_rate_max):
				self.command.angular.z = np.sign(self.command.angular.z)*self.yaw_rate_max
			self.pub1.publish(self.command)
			self.publishLookahead()
			self.pub3.publish(self.lookahead_point)
		return

if __name__ == '__main__':
	controller = guidance_controller()
	try:
		controller.start()
	except rospy.ROSInterruptException:
		pass
