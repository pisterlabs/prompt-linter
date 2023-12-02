#!/usr/bin/env python
import sys
import numpy as np
import guidance
import bezier
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import LinkStates
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

		self.position_updated = 1
		return

	def getPath(self, data): # Path subscriber callback function
		new_path = np.empty((len(data.poses), 3))
		for i in range(0, len(data.poses)):
			new_path[i, 0] = data.poses[i].pose.position.x
			new_path[i, 1] = data.poses[i].pose.position.y
			new_path[i, 2] = data.poses[i].pose.position.z
		self.path = new_path
		print('path received of size: %d' % self.path.shape[0])
		self.fitPath()
		self.path_updated = 1
		return

	def getGoalPose(self, data): # Goal Pose subscriber callback function
		q = Quaternion()
		q = data.pose.orientation
		self.goal_yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
		# print("Goal pose yaw is %0.2f" % ((180.0/np.pi)*self.goal_yaw))
		return

	def publishCrossTrack(self):
		# Remove the old marker
		self.CT_marker.action = 2
		self.pub2.publish(self.CT_marker)

		# Add a new one
		self.CT_marker.action = 0
		self.CT_marker.points[0] = self.position
		self.CT_marker.points[1] = self.CT
		self.pub2.publish(self.CT_marker)
		return

	def publishPathBezier(self, N):
		if (self.path.shape[1] < 1):
			return
		if (self.path_fit == 0):
			return
		new_path = Path()
		new_path.header.frame_id = self.fixed_frame
		new_path.header.stamp = rospy.get_rostime()
		points = bezier.getCurve(self.path_bezier, N)
		for i in range(0,N):
			new_pose_stamped = PoseStamped()
			new_pose_stamped.pose.position.x = points[i,0]
			new_pose_stamped.pose.position.y = points[i,1]
			new_pose_stamped.pose.position.z = points[i,2]
			new_path.poses.append(new_pose_stamped)
		self.pub4.publish(new_path)
		return

	def fitPath(self):
		n = 10
		d = self.path.shape[0]/n
		fit_path = np.empty((n+2, 3))
		for i in range(1, n+1):
			fit_path[i, :] = self.path[d*i-1, :]
		fit_path[0, :] =  self.path[0, :]
		fit_path[-1, :] =  self.path[-1, :]
		# print(fit_path)
		self.path_bezier = bezier.fitCurveCubic(fit_path)
		self.path_fit = 1
		return

	def updateCommand(self): # Updates the twist command for publishing
		# Check if the subscribers have updated the robot position and path
		if (self.path.shape[1] < 1):
			print("No guidance command, path is empty.")
			return

		if (self.position_updated == 0):
			print("No odometry message.")
			return

		# Convert the body frame x-axis of the robot to inertial frame
		heading_body = np.array([[1.0], [0.0], [0.0]])
		heading_inertial = np.matmul(self.R, heading_body)
		velocity_inertial = self.speed*np.array([heading_inertial[0,0], heading_inertial[1,0], heading_inertial[2,0]])

		# Find the crosstrack point for the guidance controller
		# Store the vehicle position for now
		p_robot = np.array([self.position.x, self.position.y, self.position.z])

		# Get crosstrack point from robot position to bezier path
		if (self.path_fit == 0):
			return
		crosstrack_t = bezier.projectPointOntoCurve(self.path_bezier, p_robot, 0.001) # parametric coordinate of crosstrack point
		crosstrack_array = np.squeeze(bezier.getBezier(self.path_bezier, crosstrack_t))
		self.CT.x = crosstrack_array[0]
		self.CT.y = crosstrack_array[1]
		self.CT.z = crosstrack_array[2]
		self.crosstrack_point.point = self.CT

		# Generate a feedforward turn command from the curvature at the crosstrack point, chi_dot = Velocity/Radius
		if self.vehicle_type == "ground":
			chi_dot_feedforward = self.speed*bezier.getCurvature2D(self.path_bezier, crosstrack_t)
			crosstrack_array[2] = p_robot[2]
		else:
			chi_dot_feedforward = self.speed*bezier.getCurvature3D(self.path_bezier, crosstrack_t)

		# Use a stanley controller to generate a feedback command
		# All variable names refer to the Hoffmann-Stanley '07 paper
		k = 1.0
		k_ab = 0.8
		v = self.speed # velocity
		crosstrack_tangent = bezier.getTangent(self.path_bezier, crosstrack_t)
		crosstrack_tangent_angle = np.arctan2(crosstrack_tangent[1], crosstrack_tangent[0])
		psi = (np.pi/180.0)*guidance.angle_Diff((180.0/np.pi)*self.yaw, (180.0/np.pi)*crosstrack_tangent_angle)
		if self.vehicle_type == "ground":
			crosstrack_tangent[2] = 0
		crosstrack_error_vector = p_robot - crosstrack_array
		crosstrack_error_vector = crosstrack_error_vector - np.dot(crosstrack_tangent, crosstrack_error_vector)*crosstrack_tangent
		crosstrack_angle = np.arctan2(crosstrack_error_vector[1], crosstrack_error_vector[0])
		e = np.linalg.norm(crosstrack_error_vector)*np.sign(guidance.angle_Diff((180.0/np.pi)*crosstrack_angle, (180.0/np.pi)*self.yaw)) # crosstrack error
		# print('psi = %0.2f deg' % ((180.0/np.pi)*psi))
		# print('crosstrack angle = %0.2f deg' % ((180.0/np.pi)*crosstrack_angle))
		# print('e = %0.2f' % (e))
		delta = psi + np.arctan(k*e/v)
		chi_dot_feedback = -k_ab*v*np.sin(delta)

		# Velocity command
		self.command.linear.x = self.speed
		if self.feedforward_On:
			self.command.angular.z = chi_dot_feedforward + chi_dot_feedback
		else:
			self.command.angular.z = chi_dot_feedback
		# self.command.angular.z = chi_dot_feedback
		# self.command.angular.z = chi_dot_feedforward

		return

	def __init__(self):
		# Booleans for first subscription receive
		self.position_updated = 0
		self.path_updated = 0
		self.path_fit = 0

		# Initialize ROS node and Subscribers
		node_name = 'bezier_guidance_controller'
		rospy.init_node(node_name)
		rospy.Subscriber('odometry', Odometry, self.getPosition)
		self.link_id = -1
		self.path = np.empty((3,0))
		rospy.Subscriber('path', Path, self.getPath)
		rospy.Subscriber('goal_pose', PoseStamped, self.getGoalPose)
		self.goal_yaw = 0.0
		self.R = np.zeros((3,3))

		# Set controller specific parameters
		self.vehicle_type = rospy.get_param("guidance_bezier/vehicle_type", "ground") # vehicle type (ground vs air)
		self.speed = float(rospy.get_param("guidance_bezier/speed", 1.0)) # m/s
		self.fixed_frame = rospy.get_param("guidance_bezier/fixed_frame", "world")
		self.feedforward_On = rospy.get_param("guidance_bezier/feedforward", False)

		# Initialize Publisher topics
		self.pub1 = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		self.pub2 = rospy.Publisher('crosstrack_vec', Marker, queue_size=10)
		self.pub3 = rospy.Publisher('crosstrack_point', PointStamped, queue_size=10)
		self.pub4 = rospy.Publisher('path_bezier', Path, queue_size=10)

		self.crosstrack_point = PointStamped()
		self.crosstrack_point.header.frame_id = self.fixed_frame

		# Initialize twist object for publishing
		self.command = Twist()
		self.command.linear.x = 0.0
		self.command.linear.y = 0.0
		self.command.linear.z = 0.0
		self.command.angular.x = 0.0
		self.command.angular.y = 0.0
		self.command.angular.z = 0.0

		# Initialize Crosstrack vector for publishing
		self.CT = Point()
		self.CT.x = 0.0
		self.CT.y = 0.0
		self.CT.z = 0.0
		self.position = Point()
		self.position.x = 0.0
		self.position.y = 0.0
		self.position.z = 0.0
		self.CT_marker = Marker()
		self.CT_marker.type = 4
		self.CT_marker.header.frame_id = self.fixed_frame
		self.CT_marker.header.stamp = rospy.Time()
		self.CT_marker.id = 101;
		self.CT_marker.scale.x = 0.05
		self.CT_marker.color.b = 1.0
		self.CT_marker.color.a = 1.0
		self.CT_marker.pose.orientation.w = 1.0
		self.CT_marker.action = 0
		self.CT_marker.points.append(self.position)
		self.CT_marker.points.append(self.CT)

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
			self.publishCrossTrack()
			self.pub3.publish(self.crosstrack_point)
			self.publishPathBezier(30)
		return

if __name__ == '__main__':
	controller = guidance_controller()
	try:
		controller.start()
	except rospy.ROSInterruptException:
		pass
