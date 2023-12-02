#!/usr/bin/python3

# This shit is almost directly copied from Martin Falang (2021-2022), to have
# a reference to compare to in the project thesis

# The major difference compared to what was developed by Falang, is that the code 
# has undertaken some ROSification. Thus, actually exploiting some basic 
# ROS functionalities 

# The method is not recommended, and is currently not maintained. It performed 
# similar to the pure-pursuit guidance module. Due to pure pursuit being
# standardized, it is recommended. To preserve some of the shit Falang did, in 
# case it becomes useful for future master thesises, this is kept 


assert 0, "This code should not be used! It was only used temporary during the project thesis\
  to function as a comparative work against the trash of M. Falang. Use pure pursuit guidance, if possible!"

import rospy 
from geometry_msgs.msg import TwistStamped, PointStamped, QuaternionStamped

from anafi_uav_msgs.msg import PointWithCovarianceStamped

import numpy as np
import guidance_helpers.utilities as utilities

from scipy.spatial.transform import Rotation

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

class PIDGuidanceLaw():

  def __init__(self):

    rospy.init_node("pid_guidance_node")
    self.rate = rospy.Rate(20) # Hardcoded from the pure-pursuit guidance

    # Values are directly from the config file Martin used
    self._Kp_x = 0.5
    self._Ki_x = 0.0
    self._Kd_x = 0    

    self._Kp_y = 0.5
    self._Ki_y = 0.0
    self._Kd_y = 0    

    self._Kp_z = 0.2  
    self._Ki_z = 0.0 # 0.0    
    self._Kd_z = 0    

    self._vx_limits = (-0.3, 0.3) 
    self._vy_limits = (-0.3, 0.3) 
    self._vz_limits = (-0.1, 0.1) 

    self._prev_ts = None
    self._error_int = np.zeros(3)
    self._prev_error = np.zeros(3)

    self.position_timestamp : rospy.Time = None
    self.attitude_timestamp : rospy.Time = None

    self.last_rotation_matrix_body_to_vehicle : np.ndarray = None

    # Initialize subscribers
    self.use_ned_pos_from_gnss : bool = rospy.get_param("/use_ned_pos_from_gnss")
    if self.use_ned_pos_from_gnss:
      rospy.loginfo("Node using position estimates from GNSS. Estimates from EKF disabled")
      rospy.Subscriber("/anafi/ned_pos_from_gnss", PointStamped, self._ned_pos_cb)
      rospy.Subscriber("/anafi/attitude", QuaternionStamped, self._attitude_cb)
    else:
      rospy.loginfo("Node using position estimates from EKF. Position estimates from GNSS disabled")
      rospy.Subscriber("/estimate/ekf", PointWithCovarianceStamped, self._ekf_cb)

    # Initialize publishers
    self.reference_velocity_publisher = rospy.Publisher("/guidance/pid/velocity_reference", TwistStamped, queue_size=1)


  def _clamp(self, value: float, limits: tuple):
    return np.min([np.max([value, limits[0]]), limits[1]])


  def _ekf_cb(self, msg : PointWithCovarianceStamped) -> None:
    msg_timestamp = msg.header.stamp

    if not utilities.is_new_msg_timestamp(self.position_timestamp, msg_timestamp):
      # Old message
      return
    
    self.position_timestamp = msg_timestamp
    self.position = -np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float).reshape((3, 1)) 


  def _ned_pos_cb(self, msg : PointStamped) -> None:
    msg_timestamp = msg.header.stamp

    if not utilities.is_new_msg_timestamp(self.position_timestamp, msg_timestamp):
      # Old message
      return
    
    if self.last_rotation_matrix_body_to_vehicle is None:
      # Impossible to convert positions to body frame
      return
    
    # Positions must be transformed to body
    self.position_timestamp = msg_timestamp
    self.position = -self.last_rotation_matrix_body_to_vehicle.T @ np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float).reshape((3, 1)) 


  def _attitude_cb(self, msg : QuaternionStamped) -> None:
    msg_timestamp = msg.header.stamp

    if not utilities.is_new_msg_timestamp(self.attitude_timestamp, msg_timestamp):
      # Old message
      return
    
    self.attitude_timestamp = msg_timestamp
    rotation = Rotation.from_quat([msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w])
    self.attitude_rpy = rotation.as_euler('xyz', degrees=False).reshape((3, 1))
    self.last_rotation_matrix_body_to_vehicle = rotation.as_matrix()


  def _get_position_error(self) -> np.ndarray:
    if (self.position_timestamp is None):
      return np.zeros((3, 1))

    # Using a target-position above the helipad to guide safely
    # target_position = np.array([0, 0, -0.25])
    # altitude_error = -self.position[2] # Convert between frames
    if np.linalg.norm(self.position[:2]) > 0.5:
      altitude_error = 0 #altitude_error - 1
    else:
      altitude_error = self.position[2]
    return np.array([self.position[0], self.position[1], altitude_error], dtype=np.float) 


  def get_velocity_reference(self, pos_error_body: np.ndarray, ts: float, debug=False) -> np.ndarray:
    """
    Almost copy-paste from the code developed by Martin Falang
    """

    control3D = True # (pos_error_body.shape[0] == 3)

    e_x = pos_error_body[0]
    e_y = pos_error_body[1]

    if control3D:
      e_z = pos_error_body[2]

    if self._prev_ts is not None and ts != self._prev_ts:
      dt = (ts - self._prev_ts).to_sec()

      e_dot_x = (e_x - self._prev_error[0]) / dt
      e_dot_y = (e_y - self._prev_error[1]) / dt

      if control3D:
        e_dot_z = (e_z - self._prev_error[2]) / dt

      if control3D:
        self._prev_error = pos_error_body
      else:
        self._prev_error = np.hstack((pos_error_body, 0))

      # Avoid integral windup
      if self._vx_limits[0] <= self._error_int[0] <= self._vx_limits[1]:
        self._error_int[0] += e_x * dt

      if self._vy_limits[0] <= self._error_int[1] <= self._vy_limits[1]:
        self._error_int[1] += e_y * dt

      if control3D:
        if self._vz_limits[0] <= self._error_int[2] <= self._vz_limits[1]:
          self._error_int[2] += e_z * dt

    else:
      e_dot_x = e_dot_y = e_dot_z = 0

    self._prev_ts = ts

    vx_reference = self._Kp_x*e_x + self._Kd_x*e_dot_x + self._Ki_x*self._error_int[0]
    vy_reference = self._Kp_y*e_y + self._Kd_y*e_dot_y + self._Ki_y*self._error_int[1]

    vx_reference = self._clamp(vx_reference, self._vx_limits)
    vy_reference = self._clamp(vy_reference, self._vy_limits)

    if control3D:
      vz_reference = self._Kp_z*e_z + self._Kd_z*e_dot_z + self._Ki_z*self._error_int[2]
      vz_reference = self._clamp(vz_reference, self._vz_limits)
      velocity_reference = np.array([vx_reference, vy_reference, vz_reference], dtype=np.float)
    else:
      velocity_reference = np.array([vx_reference, vy_reference, 0], dtype=np.float)

    if debug:
      print(f"Timestamp: {ts}")
      print(f"Vx gains:\tP: {self._Kp_x*e_x:.3f}\tI: {self._Ki_x*self._error_int[0]:.3f}\tD: {self._Kd_x*e_dot_x:.3f} ")
      print(f"Vy gains:\tP: {self._Kp_y*e_y:.3f}\tI: {self._Ki_y*self._error_int[1]:.3f}\tD: {self._Kd_y*e_dot_y:.3f} ")
      if control3D:
        print(f"Vz gains:\tP: {self._Kp_z*e_z:.3f}\tI: {self._Ki_z*self._error_int[2]:.3f}\tD: {self._Kd_z*e_dot_y:.3f} ")
        print(f"Velocity references:\t vx: {vx_reference:.3f}\t vy: {vy_reference:.3f} vz: {vz_reference:.3f}")
      else:
        print(f"Velocity references:\t vx: {vx_reference:.3f}\t vy: {vy_reference:.3f}")
      print()

    return velocity_reference


  def run(self) -> None:
    while not rospy.is_shutdown():
      timestamp = rospy.Time.now()
      pos_error = self._get_position_error()

      velocity_reference = self.get_velocity_reference(pos_error_body=pos_error, ts=timestamp)

      twist_msg = TwistStamped()
      twist_msg.header.stamp = timestamp
      twist_msg.twist.linear.x = velocity_reference[0]
      twist_msg.twist.linear.y = velocity_reference[1]
      twist_msg.twist.linear.z = velocity_reference[2]

      self.reference_velocity_publisher.publish(twist_msg)

      self.rate.sleep()



def main():
  pid_guidance_law = PIDGuidanceLaw()
  pid_guidance_law.run()

if __name__ == '__main__':
  main()

