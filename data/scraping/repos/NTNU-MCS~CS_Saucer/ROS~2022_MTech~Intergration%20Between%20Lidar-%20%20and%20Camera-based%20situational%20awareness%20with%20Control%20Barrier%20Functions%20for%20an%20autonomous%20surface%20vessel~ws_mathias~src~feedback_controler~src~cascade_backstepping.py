#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# cascade_backstepping.py:
#
#    Class for a maneuvering controller using the cascade backstepping control
#    lyapunov design method. The controller utilizes a two-dimensional path
#    parameter s = [s1, s2]^T and considers positional and heading control
#    seperatly. The design is based of R. Skjetne (2021) and M. Marley (2021)
#
#    Controller()
#
#
# Methods:
#
#    [] = switch()
#         - Switches the control mode from manual joystick control to automatic
#           at the press of a DS4 button. Usually mapped to triangle.
#
#    [] = getRotations()
#         - Utility function that computes the rotation matrix R(psi) and R2(psi)
#           for the current heading state
#
#    [] = saturate()
#         - Saturates the commanded forces to avoid violent commands. Max force
#           is F_max = 1 N and max moment is T_max = 0.3 Nm
#
#    --------------------- Automatic control ---------------------
#    [] = z1()
#        - Computes the positional error state for the controller
#
#    [] = virtual_control()
#        - Computes the virtual control law alpha
#
#    [] = z2()
#        - Computes the error state z2
#
#    [] = speed_assignment_with_update_law()
#        - Updates our path speed with the speed assignment and unit tangent
#          update law
#
#    [] = virtual_control_derivatives()
#        - Computes the virtual control derivatives alpha_dot
#
#    [] = clf_control_law()
#        - Computes the commanded forces tau_cmd using the control lyapunov
#          function desing.
#
#    --------------------- Manual control ---------------------
#    [] = joystick_ctrl(lStickX, lStickY, rStickX, rStickY, R2, L2)
#        - Maps input from a DS4 joystick controller to the generalized force
#          vector tau_cmd. This enables manual control of the vessel.
#
#    --------------------------- ROS ---------------------------
#    [] = updateState(data)
#       - Callback function for the state estimations. Recieves message of
#         signals from observer module and maps them to correct class variables
#
#    [] = updateReference()
#       - Callback function for the reference signals. Recieves message of
#         signals from guidance module and maps them to correct class variables
#
#    [] = updateS()
#         - Callback function for the path parameters. Recieves message of
#           signals from guidance module and maps them to correct class variables
#
#    [] = updateGains()
#        - Callback function for the controller gains. Recieves message of
#          signals from gain server and maps them to gain variables
#
#    [] = publishTau()
#        - Publishes the commanded forces tau_cmd to the ROS-topic /CSS/tau
#
# References:
#
#    M. Solheim (2022). Intergration between lidar- and camera-based situational
#    awareness and control barrier functions for an autonomous surface vessel.
#    Master thesis. Norwegian University of Science and Technology, Norway.
#
#   M. Marley (2021). Technical Note: Maneuvering control design using two path
#   variable, Rev B. Norwegian University of Science and Technology, Norway.
#
#   R. Skjetne (2021). Technical Note: Cascade backstepping-based maneuvering
#                      control design for a low-speed fully-actuated ship
#
# Author:     Mathias N. Solheim
# Revised:    28.05.2022 by M. Solheim < Added better comments and documentation >
# Tested:     04.04.2022 by M. Solheim
# ---------------------------------------------------------------------------

import rospy
import numpy as np
import dynamic_reconfigure.client
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Joy
from messages.msg import state_estimation, guidance_signals, s
from math import sqrt
from common_tools.lib import ps4
from common_tools.math_tools import Rzyx, string2array, ssa


class Controller(object):
    def __init__(self):
        self.MODE = 1                                    # Initialize with manual control, 1 for manual, 0 for automatic
        self.M = np.diag(np.array([9.51, 9.51, 0.116]))  # Inertia matrix
        self.D = np.diag(np.array([1.96, 1.96, 0.196]))  # Damping matrix

        # Initialize gains, will be dynamically reconfigured
        self.K1 = np.diag(np.ones(3))
        self.K2 = np.diag(np.ones(3))

        # Saturation limits
        self.F_max = 1                      # Saturation on forces in Surge and Sway [N]
        self.T_max = 0.3                    # Saturation on moment in Yaw [Nm]

        # Initialize postional estimations
        self.psi = 0
        self.r = 0
        self.p = np.zeros((2, 1))
        self.v = np.zeros((2, 1))
        self.b = np.zeros((3, 1))

        # Intitialize guidance signals
        self.psi_d = 0
        self.psi_d_dot = 0
        self.psi_d_ddot = 0
        self.pd = np.zeros((2, 1))
        self.pd_s1 = np.zeros((2, 1))
        self.pd_s2 = np.zeros((2, 1))
        self.pd_ss1 = np.zeros((2, 1))
        self.pd_ss2 = np.zeros((2, 1))

        # Initial tauMsg
        self.tau = np.zeros((3, 1))

        # Matrix :)
        self.S = np.array([[0, -1], [1, 0]])

        # Speed assignment and update law
        self.w = np.zeros((2, 1))           # Update law
        self.v_s = np.zeros((2, 1))         # Speed assignment
        self.v_st = np.zeros((2, 1))        # Time derivative of speed assignment, assumed zero
        self.v_ss1 = np.zeros((2, 1))       # Derivative with respect to s1
        self.v_ss2 = np.zeros((2, 1))       # Derivative with respect to s2

        self.s = np.zeros((2, 1))           # Path parameters
        self.s_dot = np.zeros((2, 1))       # Path speeds

        self.mu1 = 0                        # Update law thing1
        self.mu2 = 0                        # Update law thing2

        # Errors and virtual controls :)
        self.z1_p = np.zeros((2, 1))        # Positional error
        self.z1_psi = 0
        self.z1 = np.zeros((3, 1))
        self.z2_p = np.zeros((2, 1))        # ? error (I am an awful student lmao)
        self.z2_psi = 0
        self.z2 = np.zeros((3, 1))
        self.alpha_p = np.zeros((2, 1))     # Virtual control
        self.alpha_psi = 0
        self.alpha = np.zeros((3, 1))
        self.alpha_dot = np.zeros((3, 1))   # Derivative of virtual control
        self.alpha_dot_p = np.zeros((2, 1))
        self.alpha_dot_psi = 0
        # Rotation matrices
        self.R = Rzyx(self.psi)
        self.R2 = self.R[0:2, 0:2]

        # Initialize ROS-publisher for the commanded forces
        self.pubTau = rospy.Publisher('CSS/tau', Float64MultiArray, queue_size=1)  # Publisher
        self.tauMsg = Float64MultiArray()                                          # Message is 1x3 vector!

    def switch(self, triangle):
        """
        Switches the control-mode between automatic and manual
        """
        if triangle:                       # If triangle is pressed switch mode
            if self.MODE == 1:             # Check if manual, and switch to auto
                self.MODE = 2
                rospy.loginfo('Entering automatic control mode')
            else:                          # If not manual, then switch to it
                self.MODE = 1
                rospy.loginfo('Entering manual control mode')

    def compute_z1(self):
        """
        Computes z1 error state
        """
        self.z1_p = self.R2.T@(self.p - self.pd)
        self.z1_psi = ssa(self.psi - self.psi_d)
        self.z1[0:2, 0:1] = self.z1_p
        self.z1[2, 0] = self.z1_psi

    def virtual_control(self):
        """
        Computes the virtual control
        """
        self.alpha_p = -self.K1[0:2, 0:2]@self.z1_p + self.R2.T@self.pd_s1*self.v_s[0] + self.R2.T@self.pd_s2*self.v_s[1]
        self.alpha_psi = -self.K1[2, 2]*self.z1_psi + self.psi_d_dot
        self.alpha[0:2, 0:1] = self.alpha_p
        self.alpha[2, 0] = self.alpha_psi

    def compute_z2(self):
        """
        Computes the z2 error state
        """
        self.z2_p = self.v - self.alpha_p
        self.z2_psi = self.r - self.alpha_psi
        self.z2[0:2, 0:1] = self.z2_p
        self.z2[2, 0] = self.z2_psi

    def speed_assignment_with_update_law(self):
        w1 = (self.mu1/np.linalg.norm(self.pd_s1))*self.pd_s1.T@self.R2@self.z1_p
        w2 = (self.mu2/np.linalg.norm(self.pd_s2))*self.pd_s2.T@self.R2@self.z1_p
        self.w = np.array([[w1], [w2]])
        self.s_dot = self.v_s + self.w

    def getRotations(self):
        self.R = Rzyx(self.psi)
        self.R2 = self.R[0:2, 0:2]

    def virtual_control_derivatives(self):
        """
        Computes the virtual control derivatives for step 2 of the
        cascade design.
        """
        self.alpha_dot_p = self.K1[0:2, 0:2]@(self.r*self.S)@self.z1_p - self.K1[0:2, 0:2]@self.v + \
                           self.K1[0:2, 0:2]@self.R2.T@(self.pd_s1*self.s_dot[0, 0] + self.pd_s2*self.s_dot[1, 0]) - \
                           (self.r*self.S)@self.R2.T@(self.pd_s1*self.v_s[0, 0] + self.pd_s2*self.v_s[1, 0]) + \
                           self.R2.T@(self.pd_ss1*self.v_s[0, 0]*self.s_dot[0, 0] + self.pd_s1*self.v_ss1[0, 0]*self.s_dot[0, 0] +
                           self.pd_s1*self.v_ss2[0, 0]*self.s_dot[1, 0] + self.pd_s1*self.v_st[0, 0]) + \
                           self.R2.T@(self.pd_ss2*self.v_s[1]*self.s_dot[1, 0] + self.pd_s2*self.v_ss1[1, 0]*self.s_dot[1, 0] +
                           self.pd_s2*self.v_ss2[1, 0]*self.s_dot[1, 0] + self.pd_s1*self.v_st[1, 0])
        self.alpha_dot_psi = -self.K1[2, 2]*(self.r - self.psi_d_dot) + self.psi_d_ddot
        self.alpha_dot[0:2, 0:1] = self.alpha_dot_p
        self.alpha_dot[2, 0] = self.alpha_dot_psi


    def clf_control_law(self):
        self.tau = (-self.K2@self.z2 - self.R.T@self.b + self.D@self.alpha + self.M@self.alpha_dot)


    def joystick_ctrl(self, lStickX, lStickY, rStickX, rStickY, R2, L2):
        """
        Maps the input from a Dualshock 4 controller to a generalized
        force vector i BODY-frame.
        """

        X = (lStickY + rStickY)  # Surge
        Y = (lStickX + rStickX)  # Sway
        N = (R2 - L2)            # Yaw

        self.tau = np.array([[X], [Y], [N]])

    # def fld_control_law(self):
    #    self.tau = self.D@v - self.M@(self.K2@self.z2 + self.b - self.alpha_dot)

    def saturate(self):
        """
        Saturates the commanded force to the vessel
        """
        if (self.tau[0, 0] == 0 and self.tau[1, 0] == 0):
            ck = self.F_max/(sqrt(self.tau[0, 0]**2 + self.tau[1, 0]**2 + 0.00001))
        else:
            ck = self.F_max/sqrt(self.tau[0, 0]**2 + self.tau[1, 0]**2 + 0.00001)

        # Saturate surge and sway
        if (ck < 1):
            self.tau[0, 0] = ck*self.tau[0, 0]
            self.tau[1, 0] = ck*self.tau[1, 0]

        # Saturate yawlaw
        if (np.abs(self.tau[2, 0]) >= self.T_max):
            self.tau[2, 0] = np.sign(self.tau[2, 0])*self.T_max

    def updateState(self, data):
        """
        Callback function that updates the state estimation variables of the
        controller with the signals from the guidance module
        """
        self.psi = data.eta_hat[2]
        self.r = data.nu_hat[2]
        self.bias_hat = data.bias_hat
        self.p = np.resize(np.array(data.eta_hat[0:2]), (2, 1))
        self.v = np.resize(np.array(data.nu_hat[0:2]), (2, 1))
        self.b = np.resize(np.array(data.bias_hat), (3, 1))

    def updateReference(self, data):
        """
        Callback function that updates the reference variables with signals from
        the observer module
        """
        # Reference signals
        self.psi_d = data.eta_d[2]
        self.psi_d_dot = data.psi_d_dot
        self.psi_d_ddot = data.psi_d_ddot
        self.pd = np.resize(np.array(data.eta_d[0:2]), (2, 1))
        self.pd_s1 = np.resize(np.array(data.pd_s1), (2, 1))
        self.pd_s2 = np.resize(np.array(data.pd_s2), (2, 1))
        self.pd_ss1 = np.resize(np.array(data.pd_ss1), (2, 1))
        self.pd_ss2 = np.resize(np.array(data.pd_ss2), (2, 1))

    def updateS(self, data):
        """
        Callback function that updates the path variable and its derivatives with
        signals from the guidance module
        """
        self.s = np.resize(data.s, (2, 1))
        self.v_s = np.resize(data.s_dot, (2, 1))

    def updateGains(self, config):
        self.K1 = np.diag(string2array(config.K1))
        self.K2 = np.diag(string2array(config.K2))
        self.mu1 = config.mu

    def publishTau(self):
        """
        Publishes the computed tau to the /CSS/tau ROS-topic
        """
        tau_data = self.tau.flatten()
        self.tauMsg.data = tau_data
        self.pubTau.publish(self.tauMsg)


if __name__ == '__main__':
    rospy.init_node('Controller')
    rospy.loginfo('Control module initialized')
    r = rospy.Rate(50)
    controller = Controller()
    gain_client = dynamic_reconfigure.client.Client('gain_server', timeout=30, config_callback=controller.updateGains)
    z1_msg = Float64MultiArray()
    rospy.Subscriber("joy", Joy, ps4.updateState)                               # Initialize a Subscriber to the /joy topic
    rospy.Subscriber("CSS/observer", state_estimation, controller.updateState)  # Initialize a Subscriber to the /CSS/observer topic
    rospy.Subscriber("CSS/reference", guidance_signals, controller.updateReference)    # Initialize a Subscriber to the /CSS/reference topic
    rospy.Subscriber("CSS/s", s, controller.updateS)                            # Initialize a Subscriber to the /CSS/s topic
    while not rospy.is_shutdown():
        controller.switch(ps4.triangle)  # Check if the user switches control mode
        if controller.MODE == 1:          # Manual control loop
            controller.joystick_ctrl(ps4.lStickX, ps4.lStickY, ps4.rStickX, ps4.rStickY, ps4.R2, ps4.L2)
            controller.saturate()
            controller.publishTau()
        elif controller.MODE == 2:      # Automatic nominal control loop
            controller.getRotations()
            controller.compute_z1()
            controller.virtual_control()
            controller.compute_z2()
            controller.speed_assignment_with_update_law()
            controller.virtual_control_derivatives()
            controller.clf_control_law()
            controller.saturate()
            controller.publishTau()
        r.sleep()
    rospy.spin()
    rospy.shutdown()
