#!/usr/bin/python3

import rospy 
import std_msgs.msg
import sensor_msgs.msg

from scipy.spatial.transform import Rotation

from geometry_msgs.msg import TwistStamped, PointStamped, QuaternionStamped, PoseWithCovarianceStamped

from anafi_uav_msgs.msg import PointWithCovarianceStamped
from anafi_uav_msgs.srv import SetDesiredPosition, SetDesiredPositionRequest, SetDesiredPositionResponse

import numpy as np
import guidance_helpers.utilities as utilities

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 


class PurePursuitGuidanceLaw():
  """
  Guidance law generating the desired velocity based on the 
  desired and current position 
  """
  def __init__(self) -> None:
    rospy.init_node("pure_pursuit_guidance_node")

    controller_rate = rospy.get_param("~node_rate", default = 20)
    self.dt = 1.0 / controller_rate 
    self.rate = rospy.Rate(controller_rate)

    # Initialize parameters
    pure_pursuit_params = rospy.get_param("~pure_pursuit_parameters")
    velocity_limits = rospy.get_param("~velocity_limits")
    
    self.ua_max = pure_pursuit_params["ua_max"]
    self.lookahead = pure_pursuit_params["lookahead"]
    self.fixed_kappa = pure_pursuit_params["kappa"]

    self.vx_limits = velocity_limits["vx"]
    self.vy_limits = velocity_limits["vy"]
    self.vz_limits = velocity_limits["vz"]

    self.desired_altitude : float = -1.0

    self.position_timestamp : std_msgs.msg.Time = None
    self.attitude_timestamp : std_msgs.msg.Time = None
    self.desired_position_timestamp : std_msgs.msg.Time = None

    self.desired_position_ned : np.ndarray = np.zeros((3, 1))  # [xd, yd, zd]
    self.position_body : np.ndarray = None 

    self.last_rotation_matrix_body_to_vehicle : np.ndarray = None

    # Set up subscribers 
    rospy.Subscriber("/guidance/desired_ned_position", PointStamped, self._desired_ned_pos_cb)
    rospy.Subscriber("/anafi/attitude", QuaternionStamped, self._attitude_cb)

    self.use_ned_pos_from_gnss : bool = rospy.get_param("/use_ned_pos_from_gnss")
    if self.use_ned_pos_from_gnss:
      rospy.loginfo("Pure pursuit using position estimates from GNSS. Estimates from EKF disabled")
      rospy.Subscriber("/anafi/ned_pos_from_gnss", PointStamped, self._ned_pos_cb)
    else:
      rospy.loginfo("Pure pursuit using position estimates from EKF. Position estimates from GNSS disabled")
      rospy.Subscriber("/estimate/ekf", PoseWithCovarianceStamped, self._ekf_cb)

    # Set up publishers
    self.reference_velocity_publisher = rospy.Publisher("/guidance/pure_pursuit/velocity_reference", TwistStamped, queue_size=1)


  def _ekf_cb(self, msg : PoseWithCovarianceStamped) -> None:
    """
    Callback setting the current poisition from the EKF estimate. Note that the position
    estimate is in body, and it is drone to helipad (origin). Thus, to get origin to drone,
    the values are negated 
    """
    msg_timestamp = msg.header.stamp

    if not utilities.is_new_msg_timestamp(self.position_timestamp, msg_timestamp):
      # Old message
      return
    
    self.position_timestamp = msg_timestamp
    self.position_body = -np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=float).reshape((3, 1)) 


  def _desired_ned_pos_cb(self, msg : PointStamped) -> None:
    """
    Callback setting the desired position for the guidance to track. 
    The target position is assumed in NED 
    """
    msg_timestamp = msg.header.stamp

    if not utilities.is_new_msg_timestamp(self.desired_position_timestamp, msg_timestamp):
      # Old message
      return
    
    self.desired_position_timestamp = msg_timestamp
    self.desired_position_ned = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float).reshape((3, 1)) 


  def _ned_pos_cb(self, msg : PointStamped) -> None:
    """
    Position estimates using the direct bridge-estimates in NED. These measurements are 
    origin to drone position
    """
    msg_timestamp = msg.header.stamp

    if not utilities.is_new_msg_timestamp(self.position_timestamp, msg_timestamp):
      # Old message
      return
    
    if self.last_rotation_matrix_body_to_vehicle is None:
      # Impossible to convert positions to body frame
      return
    
    # Positions must be transformed to body
    self.position_timestamp = msg_timestamp
    self.position_body = self.last_rotation_matrix_body_to_vehicle.T @ np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float).reshape((3, 1)) 


  def _attitude_cb(self, msg : QuaternionStamped) -> None:
    msg_timestamp = msg.header.stamp

    if not utilities.is_new_msg_timestamp(self.attitude_timestamp, msg_timestamp):
      # Old message
      return
    
    self.attitude_timestamp = msg_timestamp
    rotation = Rotation.from_quat([msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w])
    self.attitude_rpy = rotation.as_euler('xyz', degrees=False).reshape((3, 1))
    self.last_rotation_matrix_body_to_vehicle = rotation.as_matrix()


  def _clamp(
        self, 
        value: float, 
        limits: tuple
      ) -> float:
    return np.min([np.max([value, limits[0]]), limits[1]]) 


  def _get_pos_error_body(self) -> np.ndarray:
    """
    Calculates a position error in body
    Assumes the attitude is known relatively correctly, such that body can be converted
    to NED
    """
    if (self.position_timestamp is None):
      return np.zeros((3, 1))

    pos_error_ned = self.last_rotation_matrix_body_to_vehicle @ self.position_body - self.desired_position_ned
    pos_error_body = self.last_rotation_matrix_body_to_vehicle.T @ pos_error_ned

    return pos_error_body


  def calculate_velocity_reference(self) -> None:
    """
    Generate a velocity reference from a position error using the pure
    pursuit guidance law as defined in Fossen 2021.
    """
    twist_ref_msg = TwistStamped()

    vel_target = np.zeros((3, 1)) # Possible extension to use constant bearing guidance in the future

    while not rospy.is_shutdown():
      if self.position_body is None:
        self.rate.sleep()
        continue

      pos_error = self._get_pos_error_body()
      pos_error_normed = np.linalg.norm(pos_error)

      if pos_error_normed > 1e-3:
        kappa = (pos_error_normed * self.ua_max) / (np.sqrt(pos_error_normed + self.lookahead**2))
        vel_ref_unclamped = vel_target - (kappa * pos_error) / (pos_error_normed) 
      else:
        vel_ref_unclamped = np.zeros((3, 1)).ravel()

      vel_ref_x = self._clamp(vel_ref_unclamped[0], self.vx_limits)
      vel_ref_y = self._clamp(vel_ref_unclamped[1], self.vy_limits)
      vel_ref_z = self._clamp(vel_ref_unclamped[2], self.vz_limits)

      twist_ref_msg.header.stamp = rospy.Time.now()
      twist_ref_msg.twist.linear.x = vel_ref_x
      twist_ref_msg.twist.linear.y = vel_ref_y
      twist_ref_msg.twist.linear.z = vel_ref_z

      self.reference_velocity_publisher.publish(twist_ref_msg)
      self.rate.sleep()


def main():
  guidance_law = PurePursuitGuidanceLaw()
  guidance_law.calculate_velocity_reference()


if __name__ == "__main__":
  main()
