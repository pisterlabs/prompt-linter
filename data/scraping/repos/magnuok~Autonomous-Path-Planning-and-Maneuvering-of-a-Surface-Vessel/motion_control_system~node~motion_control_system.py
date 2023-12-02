#!/usr/bin/python
"""
Motion control system. Implements the maneuvering controller.

uthor: Magnus Knaedal
Date: 10.06.2020

"""
import rospy
import math
import os
import sys
import numpy as np
import random
import cProfile
import pstats
import io
import copy
import matplotlib.pyplot as plt
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseStamped, Twist, Wrench, Vector3
from guidance_system.msg import HybridPathSignal
from dynamic_reconfigure.server import Server
from dp_controller.cfg import McsTuningConfig

class MotionControllerSystem():
    """Class for motion control system
    """    

    def __init__(self):
        """Initialize motion control system
        """
        self.init_system()

        ### Ros node, srvs, pubs and subs ### 
        rospy.init_node('MotionControllerSystem', anonymous=True)
        rospy.Subscriber('GuidanceSystem/HybridPathSignal', HybridPathSignal, self.hybrid_signal_listener)
        rospy.Subscriber('/observer/eta/ned', Twist, self.eta_listener)
        rospy.Subscriber('observer/nu/body', Twist, self.nu_listener)
        rospy.Subscriber('observer/bias/acceleration', Vector3, self.bias_listener)
        
        self.pub_tau = rospy.Publisher('tau_controller', Wrench, queue_size=1)

        srv = Server(McsTuningConfig, self.parameter_callback)

    def init_system(self):
        """ 
        Initialize system matrices, parameters, and tuning variables.

        """
        self.eta = np.zeros((1,3))
        self.nu = np.zeros((1,3))
        self.bias = np.zeros((1,3))
        
        I = np.eye(3)

        K_1 = 0.1*np.diag([0.5, 0.1, 0.7])
        kappa = 0.0
        self.K_1_tilde = K_1 + kappa*I

        self.K_2 = 0*np.diag([0.1, 0.05, 0.1])

        self.tau_max = np.array([[41.0],[50.0],[55.0]])

        # Physical constants (ref Alfheim and Muggerud)
        vol     = 0.26772994
        self.m = 257.0
        self.xg = 0.0
        m       = 257.0
        rho_sea = 1025.0
        g       = 9.807
        lpp     = 3.0
        xg      = 0.0
        V_Iz = 0.31002165
        Iz = (m / vol) * V_Iz

        self.M_RB = np.matrix([[m, 0.0, 0.0], [0.0, m, m * xg], [0.0, m * xg, Iz]])
        self.M_A = np.matrix([[0.025253 * rho_sea * vol, 0.0, 0.0],
                         [0, 0.18016 * rho_sea * vol, 0.0085114 * rho_sea * vol * lpp],
                         [0, 0.0085367 * rho_sea * vol * lpp, 0.0099408 * rho_sea *
                          vol * lpp * lpp]])
        self.M = self.M_RB + self.M_A

        self.D = np.matrix([[0.1021 * rho_sea * vol * np.sqrt(g / lpp), 0.0, 0.0],
                       [0.0, 1.2122 * rho_sea * vol * np.sqrt(g / lpp),
                        0.055793 * rho_sea * vol * np.sqrt(g * lpp)],
                       [0.0, 0.055825 * rho_sea * vol * np.sqrt(g * lpp),
                        0.060053 * rho_sea * vol * lpp * np.sqrt(g * lpp)]])
           
    def control_law(self, w, v_ref, dt_v_ref, dtheta_v_ref, eta_d, dtheta_eta_d, ddtheta_eta_d):
        """The maneuvering controller. Calculates generalized desired forces.

        Arguments: See thesis/Skjetne 2005.
            w {[type]} -- [description]
            v_ref {[type]} -- [description]
            dt_v_ref {[type]} -- [description]
            dtheta_v_ref {[type]} -- [description]
            eta_d {[type]} -- [description]
            dtheta_eta_d {[type]} -- [description]
            ddtheta_eta_d {[type]} -- [description]
        """        
        
        C = self.calculate_coriolis_matrix(self.nu)
        _, R_trps = self.rotation_matrix(self.eta[2, 0])
        S = self.skew_matrix(self.nu[2, 0])

        # Wrap angle
        eta_error = self.eta - eta_d
        eta_error[2, 0] = self.ssa(eta_error[2, 0])

        # Control law functions
        z1 = np.matmul(R_trps, eta_error)

        alpha_1 = -np.matmul(self.K_1_tilde, z1) + np.matmul(R_trps, dtheta_eta_d) * v_ref

        z2 = self.nu - alpha_1

        omega_1 = np.matmul(self.K_1_tilde, np.matmul(S, z1)) - np.matmul(self.K_1_tilde, self.nu) - np.matmul(S, np.matmul(R_trps, dtheta_eta_d)) * v_ref + np.matmul(R_trps, dtheta_eta_d) * dt_v_ref

        dtheta_alpha_1 = np.matmul(self.K_1_tilde, np.matmul(R_trps, dtheta_eta_d)) + np.matmul(R_trps, ddtheta_eta_d) * v_ref + np.matmul(R_trps, dtheta_eta_d) * dtheta_v_ref

        # Control law
        # TODO: + or - on bias? # - self.bias .... - np.matmul(R_trps, self.bias)
        tau = -np.matmul(self.K_2, z2) - self.bias + np.matmul(self.D + C, alpha_1) + np.matmul(self.M, omega_1) + np.matmul(self.M, dtheta_alpha_1)*(v_ref + w)

        #print("Tau_d: %f, %f, %f" % (tau[0, 0], tau[1, 0], tau[2, 0]))
        # Output saturation
        if np.absolute(tau[0, 0]) > self.tau_max[0, 0] or np.absolute(tau[1, 0]) > self.tau_max[1, 0] or np.absolute(tau[2, 0]) > self.tau_max[2, 0]:
            if np.absolute(tau[0, 0]) > self.tau_max[0, 0]:
                tau[2] = np.sign(tau[2, 0]) * np.absolute(self.tau_max[0, 0] / tau[0, 0]) * np.absolute(tau[2, 0])
                tau[1] = np.sign(tau[1, 0]) * np.absolute(self.tau_max[0, 0] / tau[0, 0]) * np.absolute(tau[1, 0])
                tau[0] = np.sign(tau[0, 0]) * self.tau_max[0, 0]
            if np.absolute(tau[1, 0]) > self.tau_max[1, 0]:
                tau[2] = np.sign(tau[2, 0]) * np.absolute(self.tau_max[1, 0] / tau[1, 0]) * np.absolute(tau[2, 0])
                tau[0] = np.sign(tau[0, 0]) * np.absolute(self.tau_max[1, 0] / tau[1, 0]) * np.absolute(tau[0, 0])
                tau[1] = np.sign(tau[1, 0]) * self.tau_max[1]
            if np.absolute(tau[2, 0]) > self.tau_max[2, 0]:
                tau[1, 0] = np.sign(tau[1, 0]) * np.absolute(self.tau_max[2, 0] / tau[2, 0]) * np.absolute(tau[1, 0])
                tau[0, 0] = np.sign(tau[0, 0]) * np.absolute(self.tau_max[2, 0] / tau[2, 0]) * np.absolute(tau[0, 0])
                tau[2, 0] = np.sign(tau[2, 0]) * self.tau_max[2, 0]        
        #print("Tau_d_s: %f, %f, %f" % (tau[0, 0], tau[1, 0], tau[2, 0]))

        # Create ros msg
        tau_msg = Wrench()
        tau_msg.force.x = tau[0, 0]
        tau_msg.force.y = tau[1, 0]
        tau_msg.torque.z = tau[2, 0]
        self.pub_tau.publish(tau_msg)

    def calculate_coriolis_matrix(self, nu):
        """Calculates the coriolis centripetal matrix. 

        Arguments:
            nu {[list]} -- [velocity vector in body frame]

        Returns:
            [numpy matrix] -- [coriolis centripetal matrix]
        """        
        u = nu[0, 0]
        v = nu[1, 0]
        r = nu[2, 0]
        
        C_RB = np.matrix([[0.0, 0.0, -self.m * (self.xg * r + v)], [0.0, 0.0, self.m * u],
                          [self.m * (self.xg * r + v), -self.m * u, 0.0]])
        C_A = np.matrix([[0.0, 0.0, -self.M_A[1, 1] * v + (-self.M_A[1, 2]) * r], [0.0, 0.0, -self.M_A[0, 0] * u],
                         [self.M_A[1, 1] * v - (-self.M_A[1, 2]) * r, self.M_A[0, 0] * u, 0.0]])
        C = C_RB + C_A        

        return C
    
    ### Utils ###

    @staticmethod
    def rotation_matrix(psi):
        
        R = np.array([[math.cos(psi), -math.sin(psi), 0.0],
                        [math.sin(psi), math.cos(psi), 0.0],
                        [0.0,     0.0,      1] ])        

        R_trps = np.transpose(R)
        return R, R_trps

    @staticmethod
    def skew_matrix(r):
        """Calculate the skew matrix

        Arguments:
            r {[double]} -- [yaw rate]

        Returns:
            [numpy matrix] -- [skew matrix]
        """        
        
        S = np.array([[0.0,   -r,     0.0],
                    [r,     0.0,      0.0],
                    [0.0,     0.0,      0.0] ])        

        return S

    @staticmethod
    def ssa(angle):
        """
        Smallest signed angle. Maps angle into interval [-pi pi]
        """
        wrpd_angle = (angle + math.pi) % (2.0*math.pi) - math.pi
        return wrpd_angle
    
    ### Callbacks ###
    def hybrid_signal_listener(self, signal):

        w = signal.w_ref
        v_ref = signal.v_ref
        dt_v_ref = signal.dt_v
        dtheta_v_ref = signal.dtheta_v
        
        eta_d = np.array([[signal.eta_d.x],
                               [signal.eta_d.y],
                               [signal.eta_d.theta] ])

        dtheta_eta_d = np.array([[signal.dot_eta_d.x],
                               [signal.dot_eta_d.y],
                               [signal.dot_eta_d.theta] ])

        ddtheta_eta_d = np.array([[signal.ddot_eta_d.x],
                               [signal.ddot_eta_d.y],
                               [signal.ddot_eta_d.theta] ])

        self.control_law(w, v_ref, dt_v_ref, dtheta_v_ref, eta_d, dtheta_eta_d, ddtheta_eta_d)

    def parameter_callback(self, config, level):
        """ Callback function for updating parameters.

        Parameters
        ----------
        config : ParameterGenerator()
            configuration parameters

        """

        self.K_1_tilde = np.diag([config.Kp_surge, config.Kp_sway, config.Kp_heading])

        self.K_2 = np.diag([config.Kd_surge, config.Kd_sway, config.Kd_heading])

        self.tau_max = np.array([[config.Tmax_surge], [config.Tmax_sway], [config.Tmax_heading]])

        rospy.loginfo('Config setup complete! ')

        return config

    def eta_listener(self, eta):
        """
        Listens to eta (NED) position.
        """
        # [x, y, psi]

        deg2rad = math.pi/180.0

        self.eta = np.array([ [eta.linear.x], [eta.linear.y], [deg2rad*eta.angular.z] ])

    def nu_listener(self, nu):
        """
        Listens to eta (NED) position.
        """
        # [x, y, psi]
        self.nu = np.array([ [nu.linear.x], [nu.linear.y], [nu.angular.z] ])

    def bias_listener(self, bias):
        """Bias callback

        Arguments:
            bias {[list]} -- [bias estimates]
        """        
        self.bias = np.array([ [bias.x], [bias.y], [bias.z] ])
     
if __name__ == '__main__':
    try:
        MotionControllerSystem()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

