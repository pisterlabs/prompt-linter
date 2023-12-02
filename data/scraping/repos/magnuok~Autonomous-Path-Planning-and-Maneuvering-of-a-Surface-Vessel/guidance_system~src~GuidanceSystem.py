#!/usr/bin/python
"""
Guidance system. Using Bezier curve. Optimal places control points of the curve according
to method developed in master thesis of Magnus Knaedal.

Two functions are performing path generation; one obtaining wps from CME, and one from local planner.

Author: Magnus Knaedal
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
from cvxopt import matrix, solvers
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseStamped, Twist
from guidance_system.msg import HybridPathSignal
from dynamic_reconfigure.server import Server
from dp_controller.cfg import GsTuningConfig

# Import utility-functions
try:
    from _plot_utils import Plot_utils
except ImportError:
    raise

class GuidanceSystem(Plot_utils):

    def __init__(self):

        self.init_parameters()

        ### Ros node, srvs, pubs and subs ### 
        rospy.init_node('GuidanceSystem', anonymous=True)
        rospy.Subscriber("CME/wp_pp", Path, self.wp_listener_cme)
        rospy.Subscriber("nav_syst/local_planner/path", Path, self.wp_listener)
        rospy.Subscriber("/observer/eta/ned", Twist, self.eta_listener)
        self.pub_hps = rospy.Publisher('GuidanceSystem/HybridPathSignal', HybridPathSignal, queue_size=1)
        self.pub_path = rospy.Publisher('GuidanceSystem/Path', Path, queue_size = 10)
        self.pub_s_dyn = rospy.Publisher('GuidanceSystem/s_dynamics', PoseStamped, queue_size = 10)

        srv = Server(GsTuningConfig, self.parameter_callback)

    def init_parameters(self):
        """Initialize parameters for Guidance System.

        """        

        ### Tuning parameters

        self.h = 0.005 # Stepsize
        self.zeta = 2 # Wall size
        self.epsilon = [self.zeta/6, self.zeta/6, self.zeta/6]

        # S: zeta = 1.5, epsilon = [self.zeta/6, self.zeta/6, self.zeta/6]
        # 8: zeta = 2, epsilon = [self.zeta/6, self.zeta/6, self.zeta/6]

        self.u_d = 0.25 # Reference Speed # Should be determined by a path planner.
        self.dt_u_d = 0.0 # Ref speed derivative
        self.mu = 0.03 # Unit tangent gradient update law tuning variable

        ### Vessel
        self.eta = np.zeros(3)

        ### Bezier curve
        self.Theta = np.arange(0, 1, self.h).tolist()
        self.n = 8 # control points
        self.Q = 0.0 # arc length
    
        ### Guidance Law
        self.i = 0 # segment counter for


        self.Pg_output = [] # List of objects containing output for each segment.
        self.s = 0 # Cont parametric var s = theta + (i-1)
        self.j = 0
        self.omega = 0
        self.om = 0
        self.theta_x = 0.7
        self.s_list = []

        ### Path generator
        self.WP = np.zeros((0, 2)) # waypoint list
        # Cost function
        self.M = self.spline_matrix(self.n)
        int_product_a_dot = 16.239
        self.W = int_product_a_dot *  np.matmul(np.transpose(self.M), self.M)
        self.W_456 = matrix(self.W[4:7, 4:7], tc='d')
        # Constraints
        # x4 <= x5 <= x6
        A1 = np.array([ [1, -1, 0],  [0, 1, -1] ])
        # Inside corridor
        A2 = np.array([ [0, 0, -1],  [0, 1, -4], [-1, 6, -12] ])
        # Bigger than x7.
        A3 = -A2
        # x1_i+1 <= x2_i+1 <= x3_i+1
        A4 = np.array([ [0, -1, 3],  [1, -5, 8] ])
        # lower bound. All greater than 0 or x7/2
        A5 = np.array([ [-1, 0, 0],  [0, -1, 0], [0, 0, -1] ])
        # Upper bound
        A6 = np.array([ [1, 0, 0],  [0, 1, 0], [0, 0, 1] ])
        self.A_opt = matrix(np.concatenate((A1, A2, A3, A4, A5, A6)), tc='d')

        # For init CP of new segment
        self.A_initCP = np.array([ [1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [-2, 0, 1, 0, 0, 0],
                                [0, -2, 0, 1, 0, 0],
                                [3, 0, -3, 0, 1, 0],
                                [0, 3, 0, -3, 0, 1] ])
        
        self.CP_prev = []
        self.first_segment = True

        ### Plotting
        self.plotlist = [] # List with plot objects        
        self.path = Path() # For plotting in rviz

    def guidance_law(self):
        """Guidance law publishing signal to motion control system.
        """
        # Wait for first segment.
        while len(self.Pg_output) == 0:
            pass

        Hz = 20.0
        T = 1.0/Hz
        rate = rospy.Rate(Hz) # Ros rate in Hz

        # Initialize variables
        s = 0.0
        v = 0.0
        i = math.floor(s) + 1.0
        theta = s - math.floor(s)

        while not rospy.is_shutdown():
            s0 = s

            # Index corresponding to theta
            idx = int(math.floor(theta/self.h))
            # First pick segment. -1 since zero index
            # Then pick idx corresponding to theta value.
            eta_d = self.Pg_output[int(i-1)].eta_d[idx]
            dot_eta_d = self.Pg_output[int(i-1)].dot_eta_d[idx]
            ddot_eta_d = self.Pg_output[int(i-1)].ddot_eta_d[idx]
            v = self.Pg_output[int(i-1)].v[idx]
            dt_v = self.Pg_output[int(i-1)].dt_v[idx]
            dtheta_v = self.Pg_output[int(i-1)].dtheta_v[idx]

            # Path parameter dynamics using unit-tangent gradient update law
            # Wrap angle
            eta_error = self.eta - eta_d
            eta_error[2] = self.ssa(eta_error[2])
            w = (self.mu / np.linalg.norm(dot_eta_d)) * np.dot( dot_eta_d, eta_error )

            # Send Hybrid path signal
            hps = self.create_hybrid_path_signal_msg(eta_d, dot_eta_d, ddot_eta_d, w, v, dt_v, dtheta_v)
            self.pub_hps.publish(hps)
            self.plot_s_dynamic(eta_d)

            # update s using Euler discretization
            s = s0 + T*(v + w)
            i = math.floor(s) + 1.0
            theta = s - math.floor(s)
            
            # Start over again. Comment out block if-sentence if not want to.
            if s >= len(self.WP)-1:
                self.Pg_output = []
                # Add current position.
                WP_list = np.copy(self.WP)

                # Rotate to start straight.
                R = self.rot_matrix(self.eta[2])
                for row in WP_list:
                    x_rot = R[0][0] * row[0] + R[0][1] * row[1]
                    y_rot = R[1][0] * row[0] + R[1][1] * row[1] 
                    row[0] = x_rot
                    row[1] = y_rot
                
                # Add current position.
                for row in WP_list:
                    row[0] += self.eta[0]
                    row[1] += self.eta[1]
    
                # run path generation
                self.first_segment = True
                self.path_generator_cme(WP_list, self.zeta)

                s0 = 0.0
                s = 0.0
                i = math.floor(s) + 1.0
                theta = s - math.floor(s)
                
            rate.sleep()

    def path_generator_cme(self, WP_list, zeta):
        """Path generator for list of wps from CME.

        Arguments:
            WP_list {[list]} -- [list of wps]
            psi_prev {[double]} -- [previous angle (inital angle)]
            zeta {[double]} -- [wall distance]
        """
        psi_prev = math.atan2(WP_list[1][1] - WP_list[0][1], WP_list[1][0] - WP_list[0][0])

        for i in range(0, WP_list.shape[0]-1):
            WP_current = WP_list[i]
            WP_next = WP_list[i+1]
            psi_next = math.atan2(WP_next[1] - WP_current[1], WP_next[0] - WP_current[0])

            if self.first_segment:
                self.CP = self.initialize_CP(WP_current, psi_prev, WP_next, psi_next, self.CP_prev, first_segment = True)
                self.first_segment = False
            else:
                self.CP = self.initialize_CP(WP_current, psi_prev, WP_next, psi_next, self.CP_prev, first_segment = False)
            
            CP_opt = self.quadratic_programming(self.CP, zeta, psi_next)

            # Calculate output of path generator for one segment
            pg_output = self.PathGenerationOutput(CP_opt, self.u_d, self.dt_u_d, self.h, self.n, self.Theta)
            self.Pg_output.append(pg_output)

            # Update s
            self.j = self.j + 1
            self.s = [theta + (self.j-1) + self.om for theta in self.Theta]
            
            self.s_list.append(self.s)

            self.plot_path(pg_output, self.path, i)

            plot_object = self.PlotOutput(WP_current, WP_next, CP_opt, zeta, pg_output, i, self.s)
            self.plotlist.append(plot_object)

            self.Q += pg_output.tot_arc_length
                
            self.CP_prev = CP_opt
            psi_prev = psi_next
        
        #self.plotting(self.plotlist)

    def path_generator(self, WP_current, psi_current, WP_next, psi_next, zeta, i):
        """Path generator for wps comming from local planner.

        Arguments:
            WP_current {[list]} -- []
            psi_current {[double]} -- [current angle]
            WP_next {[list]} -- []
            psi_next {[double]} -- [next angle]
            zeta {[double]} -- [corridor width]
            i {[int]} -- [segment number]
        """
        if self.first_segment:
            self.CP = self.initialize_CP(WP_current, psi_current, WP_next, psi_next, self.CP_prev, first_segment = True)
            self.first_segment = False
        else:
            self.CP = self.initialize_CP(WP_current, psi_current, WP_next, psi_next, self.CP_prev, first_segment = False)
        
        CP_opt = self.quadratic_programming(self.CP, zeta, psi_next)

        # Calculate output of path generator for one segment
        pg_output = self.PathGenerationOutput(CP_opt, self.u_d, self.dt_u_d, self.h, self.n, self.Theta)
        self.Pg_output.append(pg_output)

        # Update s
        self.j = self.j + 1
        self.s = [theta + (self.j-1) + self.om for theta in self.Theta]
        
        self.s_list.append(self.s)

        self.plot_path(pg_output, self.path, i)

        plot_object = self.PlotOutput(WP_current, WP_next, CP_opt, zeta, pg_output, i, self.s)
        self.plotlist.append(plot_object)

        self.Q += pg_output.tot_arc_length
            
        self.CP_prev = CP_opt
        psi_prev = psi_next
        
        #self.plotting(self.plotlist)

    def initialize_CP(self, WP_current, psi_prev, WP_next, psi_next, CP_prev, first_segment = False):
        """
         Calculate the first 4 control points for current segment and initailize
         the matrix of control points.
        
         Input:
         - WP_current: current WP
         - psi_prev: previous heading
         - WP_next: next WP
         - psi_next: next heading
         - i: Segment number.
         - CP_prev: Control points from previous segment
        
         Output:
         - CP: Initial placement of control points for current segment.
        
         CP = init_cp(WP_current, WP_next, psi_next, i, CP_prev) calculates
         control points P0 to P3, and initilize control point list.
        
         Magnus Knaedal 
        """
        # Scaling factor for initializing points
        delta_max = np.linalg.norm(WP_next - WP_current)/2.0
        a = np.array( [math.cos(psi_prev), math.sin(psi_prev)] )
        b = np.array( [math.cos(psi_next), math.sin(psi_next)] )
        if first_segment:

            return np.array([ WP_current,
                WP_current + delta_max*a/6.0,
                WP_current + delta_max*a/3.0,
                WP_current + delta_max*a,
                WP_next - delta_max * b, 
                WP_next - delta_max * b/3.0,         
                WP_next - delta_max * b/6.0, 
                WP_next ])

        else:
            
            B = np.array( [ 2*CP_prev[7][0]-CP_prev[6][0],
            2*CP_prev[7][1]-CP_prev[6][1],
            -2*CP_prev[6][0]+CP_prev[5][0],
            -2*CP_prev[6][1]+CP_prev[5][1],
            2*CP_prev[7][0]-3*CP_prev[6][0]+3*CP_prev[5][0]-CP_prev[4][0],
            2*CP_prev[7][1]-3*CP_prev[6][1]+3*CP_prev[5][1]-CP_prev[4][1] ] )

            x = np.linalg.solve(self.A_initCP, B)
            # Note: P4, P5, P6 is not necassary to calculate, since found by optimization algorithm.
            return np.array([ WP_current,
                np.array([x[0], x[1]]),
                np.array([x[2], x[3]]),
                np.array([x[4], x[5]]),
                WP_next - delta_max * b, 
                WP_next - delta_max * b/3.0,         
                WP_next - delta_max * b/6.0, 
                WP_next ])

    def quadratic_programming(self, CP, zeta, psi_next):
        """Quad prog solver for path generator

        Arguments:
            CP {[matrix]} -- [initial control points]
            zeta {[type]} -- [corridor width]
            psi_next {[type]} -- [next angle]

        Returns:
            [matrix] -- [Optimal placed control points]
        """
        R = self.rot_matrix(psi_next)
        CP_path = np.zeros((CP.shape[0], CP.shape[1]))
        origo = CP[0]
        for i in range(0, CP.shape[0]):
            CP_path[i] = CP[i] - origo
            CP_path[i] = np.matmul(np.transpose(R), np.transpose(CP_path[i]))

        x0 = CP_path[0][0]
        x1 = CP_path[1][0]
        x2 = CP_path[2][0]
        x3 = CP_path[3][0]
        x7 = CP_path[7][0]

        q4 = x0*self.W[4][0] + x1*self.W[4][1] + x2*self.W[4][2] + x3*self.W[4][3] + x7*self.W[4][7] + x0*self.W[0][4] + x1*self.W[1][4] + x2*self.W[2][4] + x3*self.W[3][4] + x7*self.W[7][4]
        q5 = x0*self.W[5][0] + x1*self.W[5][1] + x2*self.W[5][2] + x3*self.W[5][3] + x7*self.W[5][7] + x0*self.W[0][5] + x1*self.W[1][5] + x2*self.W[2][5] + x3*self.W[3][5] + x7*self.W[7][5]
        q6 = x0*self.W[6][0] + x1*self.W[6][1] + x2*self.W[6][2] + x3*self.W[6][3] + x7*self.W[6][7] + x0*self.W[0][6] + x1*self.W[1][6] + x2*self.W[2][6] + x3*self.W[3][6] + x7*self.W[7][6]
        q = matrix(np.array( [q4 ,q5, q6] ), tc='d') 

        # x4 <= x5 <= x6
        b1 = np.array( [[0], [0]] )
        # Inside corridor
        b2 = np.array( [[-x7 + zeta], [-3*x7 + zeta], [-7*x7 + zeta] ]) 
        # Bigger than x7.
        b3 = -np.array( [[-x7 + self.epsilon[0]], [-3*x7 + self.epsilon[1]], [-7*x7 + self.epsilon[2]] ])
        # x1_i+1 <= x2_i+1 <= x3_i+1
        b4 = np.array( [[2*x7], [4*x7]] )
        # All greater than zero. lower bound
        b5 = np.array([ [-x7/2], [-x7/2.0], [-x7/2.0]]) # zeros(3,1)
        #b5 = np.array([ [0], [0], [0]]) # zeros(3,1)
        # Upper bound
        b6 = np.array([ [x7], [x7], [x7]])
        b = matrix(np.concatenate((b1, b2, b3, b4, b5, b6)), tc='d')

        # Quadratic solver
        solvers.options['show_progress'] = False
        solution = solvers.qp(self.W_456, q, self.A_opt, b)
        x = solution['x']

        CP_opt456 = np.array([[x[0], CP_path[4][1]],
                            [x[1], CP_path[5][1]],
                            [x[2], CP_path[6][1]] ])
        CP_opt_path = np.concatenate( (CP_path[0:4], CP_opt456, [CP_path[7]]) )
        CP_opt = np.zeros((CP.shape[0], CP.shape[1]))
        for i in range(0, CP_opt_path.shape[0]):
            CP_opt[i] = np.matmul(R, np.transpose(CP_opt_path[i]))
            CP_opt[i] = CP_opt[i] + origo
        
        return CP_opt

    ### Utils ###

    @staticmethod
    def create_hybrid_path_signal_msg(eta_d, dot_eta_d, ddot_eta_d, w, v, dt_v, dtheta_v):
        """Create signal to be sent to motin control system (ros msg).

        Arguments:
            eta_d {[list]} -- [desired pose]
            dot_eta_d {[list]} -- [dersired pose derivative]
            ddot_eta_d {[list]} -- [dersired pose double derivative]
            w {[double]} -- [Param from unit tangent grad. update law. See Skjetne 2005 e.g.]
            v {[double]} -- [Speed profile]
            dt_v {[double]} -- [Speed profile derivative time]
            dtheta_v {[double]} -- [Speed profile derivative path variable]

        Returns:
            [ros msg] -- [Hybrid path signal]
        """        
        signal = HybridPathSignal()

        signal.w_ref = w
        signal.v_ref = v
        signal.dt_v = dt_v
        signal.dtheta_v = dtheta_v
        
        signal.eta_d.x = eta_d[0]
        signal.eta_d.y = eta_d[1]
        signal.eta_d.theta = eta_d[2]

        signal.dot_eta_d.x = dot_eta_d[0]
        signal.dot_eta_d.y = dot_eta_d[1]
        signal.dot_eta_d.theta = dot_eta_d[2]

        signal.ddot_eta_d.x = ddot_eta_d[0]
        signal.ddot_eta_d.y = ddot_eta_d[1]
        signal.ddot_eta_d.theta = ddot_eta_d[2]

        return signal

    @staticmethod
    def rot_matrix(psi):
        """Rotation matrix

        Arguments:
            psi {[double]} -- [angle]

        Returns:
            [numpy matrix] -- [rot. matrix]
        """        
        return np.array([[math.cos(psi), -math.sin(psi)], 
                        [math.sin(psi), math.cos(psi)]])
    
    @staticmethod
    def ssa(angle):
        """Smallest signed angle. Maps angle into interval [-pi pi]

        Arguments:
            angle {[double]} -- [angle]

        Returns:
            [double] -- [wrapped angle]
        """        """
        
        """
        wrpd_angle = (angle + math.pi) % (2*math.pi) - math.pi
        return wrpd_angle
    
    @staticmethod
    def spline_matrix(n):
        """Create n x n spline matrix M.

        Arguments:
            n {[int]} -- [degree]

        Returns:
            [numoy matrix] -- [spline matrix M]
        """

        M = np.zeros( (n, n) )

        # Factorial function
        Sigma = []
        n_deg = n-1
        for i in range(0, n):
            sigma = math.factorial(n_deg) / ( math.factorial(i) * math.factorial(n_deg - i) )
            Sigma.append(sigma)

        for i in range(0, n):
            for j in range(0, n):
                if (i-j < 0):
                    M[i,j] = 0
                else:
                    M[i,j] = (-1)**(i-j) * Sigma[i] * math.factorial(i) / ( math.factorial(j) * math.factorial(i-j) )

        return M

    class PathGenerationOutput:
        """Calculates ouput for path generator using the Bezier curve.

        Returns:
            [list of lists] -- [path generation output for one segment]
        """
        def __init__(self, CP, u_d, dt_u_d, h, n, Theta):
            
            P_b = self.BlendingFunctions(n, Theta)
            # Derivatives
            self.p_d = P_b.B_blend.dot(CP)
            self.dot_p_d = P_b.dot_B_blend.dot(CP)
            self.ddot_p_d = P_b.ddot_B_blend.dot(CP)
            self.dddot_p_d = P_b.dddot_B_blend.dot(CP)
            
            # Outputs
            self.psi_rad = self.heading(self.dot_p_d)
            self.psi_deg = np.array([ math.degrees(psi) for psi in self.psi_rad ])
            self.dot_psi_rad = self.dot_heading(self.dot_p_d, self.ddot_p_d)
            self.ddot_psi_rad = self.ddot_heading(self.dot_p_d, self.ddot_p_d, self.dddot_p_d)

            self.eta_d = np.array([ [p_d[0], p_d[1], psi]  for p_d, psi in zip(self.p_d, self.psi_rad)])
            self.dot_eta_d = np.array([ [dot_p_d[0], dot_p_d[1], dot_psi]  for dot_p_d, dot_psi in zip(self.dot_p_d, self.dot_psi_rad)])
            self.ddot_eta_d = np.array([ [ddot_p_d[0], ddot_p_d[1], ddot_psi]  for ddot_p_d, ddot_psi in zip(self.ddot_p_d, self.ddot_psi_rad)])

            self.K = self.curvature(self.dot_p_d, self.ddot_p_d)
            self.dot_K = self.dot_curvature(self.dot_p_d, self.ddot_p_d, self.dddot_p_d)

            self.v = self.speed_profile(u_d, self.dot_p_d)
            self.dt_v = self.dot_t_speed_profile(dt_u_d, self.dot_p_d)
            self.dtheta_v = self.dot_speed_profile(u_d, dt_u_d, self.dot_p_d, self.ddot_p_d)

            self.arcs, self.tot_arc_length = self.arc_length(self.p_d)

        @staticmethod
        def arc_length(p_d):
            """
            Arc length for each point theta and total length, which is the last element in list
            """
            arc_length = np.zeros(p_d.shape[0])

            ox = p_d[0,0]
            oy = p_d[0,1]
            clen = 0
            for i in range(1, p_d.shape[0]):
                x = p_d[i,0]
                y = p_d[i,1]

                d_x = ox - x
                d_y = oy - y
                clen += math.sqrt( d_x**2 + d_y**2 )
                arc_length[i] = clen

                ox = x
                oy = y

            return arc_length, clen

        @staticmethod
        def heading(dot_p_d):
            """Calculate heading

            Arguments:
                dot_p_d {[list]} -- [derivatives]

            Returns:
                [list] -- [heading list]
            """            
            return np.array([ math.atan2(dot_p_d[i,1], dot_p_d[i,0] ) for i in range(dot_p_d.shape[0]) ])

        @staticmethod
        def dot_heading(dot_p_d, ddot_p_d):
            """Derivative of heading

            Arguments:
                dot_p_d {[list]} -- [description]
                ddot_p_d {[list]} -- [description]

            Returns:
                [list] -- [description]
            """
            dot_psi = np.zeros(dot_p_d.shape[0])

            for i in range(dot_p_d.shape[0]):
                d_x = dot_p_d[i,0]
                dd_x = ddot_p_d[i,0]
                d_y = dot_p_d[i,1]
                dd_y = ddot_p_d[i,1]
                
                dot_psi[i] = (d_x * dd_y - dd_x * d_y) / (d_x**2 + d_y**2)

            return dot_psi

        @staticmethod
        def ddot_heading(dot_p_d, ddot_p_d, dddot_p_d):
            """Second derivative of heading

            Arguments:
                dot_p_d {[list]} -- [description]
                ddot_p_d {[list]} -- [description]
                dddot_p_d {[list]} -- [description]

            Returns:
                [list] -- [description]
            """
            ddot_psi = np.zeros(dot_p_d.shape[0])

            for i in range(dot_p_d.shape[0]):
                d_x = dot_p_d[i,0]
                dd_x = ddot_p_d[i,0]
                ddd_x = dddot_p_d[i,0]
                d_y = dot_p_d[i,1]
                dd_y = ddot_p_d[i,1]
                ddd_y = dddot_p_d[i,1]
                
                ddot_psi[i] = (d_x*ddd_y - ddd_x*d_y) / (d_x**2 + d_y**2) - 2*( (d_x*dd_y - dd_x*d_y) * (d_x*dd_x + d_y*dd_y) ) / (d_x**2 + d_y**2)**2

            return ddot_psi

        @staticmethod
        def curvature(dot_p_d, ddot_p_d):
            """
             Calculates the curvator of the bezier curve
            
             Input:
             - P = control points
             - dot_p_d, ddot_p_d: blending functions
            
             Magnus Knaedal 28.12.2019
            
            """            

            K = np.zeros(dot_p_d.shape[0])
            for i in range(dot_p_d.shape[0]):
                d_x = dot_p_d[i,0]
                dd_x = ddot_p_d[i,0]
                d_y = dot_p_d[i,1]
                dd_y = ddot_p_d[i,1]
                
                K[i] = abs( d_x * dd_y - dd_x * d_y) / ((d_x**2 + d_y**2)**(1.5))

            return K

        @staticmethod
        def dot_curvature(dot_p_d, ddot_p_d, dddot_p_d):
            """
            Derivative of curvature.
            """
            dot_K = np.zeros(dot_p_d.shape[0])
            
            for i in range(dot_p_d.shape[0]):
                d_x = dot_p_d[i,0]
                dd_x = ddot_p_d[i,0]
                ddd_x = dddot_p_d[i,0]

                d_y = dot_p_d[i,1]
                dd_y = ddot_p_d[i,1]
                ddd_y = dddot_p_d[i,1]

                dot_K[i] = ( (ddd_y * d_x - ddd_x * d_y) / (d_x**2 + d_y**2)**(1.5) ) - (3 * (d_x * dd_y - dd_x * d_y) * (2 * d_x * dd_x + 2 * d_y * dd_y) ) / (2 * (d_x**2 + d_y**2)**(2.5))

            return dot_K

        @staticmethod
        def speed_profile(u_d, dot_p_d):
            """
            Speed profile v_s,i
            """
            v = np.zeros(dot_p_d.shape[0])

            for i in range(dot_p_d.shape[0]):
                d_x = dot_p_d[i,0]
                d_y = dot_p_d[i,1]

                v[i] = u_d / math.sqrt(d_x**2 + d_y**2)

            return v            

        @staticmethod
        def dot_t_speed_profile(dt_u_d, dot_p_d):
            """
            Speed profile v_s,i
            """
            dot_t_v = np.zeros(dot_p_d.shape[0])

            for i in range(dot_p_d.shape[0]):
                d_x = dot_p_d[i,0]
                d_y = dot_p_d[i,1]

                dot_t_v[i] = dt_u_d / math.sqrt(d_x**2 + d_y**2)

            return dot_t_v

        @staticmethod
        def dot_speed_profile(u_d, dt_u_d, dot_p_d, ddot_p_d):
            """
            Speed profile v_s,i
            """
            dot_v = np.zeros(dot_p_d.shape[0])

            for i in range(dot_p_d.shape[0]):
                d_x = dot_p_d[i,0]
                dd_x = ddot_p_d[i,0]
                d_y = dot_p_d[i,1]
                dd_y = ddot_p_d[i,1]

                dot_v[i] = dt_u_d / math.sqrt(d_x**2 + d_y**2) - u_d * (d_x*dd_x + d_y*dd_y) / (d_x**2 + d_y**2)**(3.0/2.0)

            return dot_v       

        def map(self, u, arc_lengths, tot_arc_length):
            """
            Mapping function to obatain arc length parametrization of the Bez curve. NOTE: not used

            u = Percentwise how far
            arc_lengths = list of arc lengths for each theta value
            tot_arc_length = total arc length of curve segment
            """
            target_length = u * tot_arc_length
            low = 0
            high = len(arc_lengths)
            index = 0

            while low < high:
                index = low + math.floor((high - low) / 2.0)
                if arc_lengths[index] < target_length:
                    low = index + 1
                else:
                    high = index
            
            if arc_lengths[index] > target_length:
                index-= 1

            length_before = arc_lengths[index]
            if length_before == target_length:
                return index / len(arc_lengths)
            else:
                return (index + (target_length - length_before) / (arc_lengths[index+1] - length_before)) / len(arc_lengths)

        def theta_arc_param(self, p_d, arc_lengths, tot_arc_length, h):
            """Finds theta for arc length parametrization

            Arguments:
                p_d {[type]} -- [description]
                arc_lengths {[type]} -- [description]
                tot_arc_length {[type]} -- [description]
                h {[type]} -- [description]

            Returns:
                [type] -- [description]
            """            

            theta_arc = np.zeros(p_d.shape[0])
            
            u = 0
            i = 0
            while u <= 1:
                theta_arc[i] = self.map(u, arc_lengths, tot_arc_length)
                i += 1
                u += h
            
            return theta_arc

        class BlendingFunctions:
            """Calculates blending function for the bezier curve.
            """            
            def __init__(self, n, Theta):

                M = self.spline_matrix(n)
                self.blend(M, n, Theta)
                
            def spline_matrix(self, n):
                """
                Create n x n spline matrix M.
                """

                M = np.zeros( (n, n) )

                # Factorial function
                Sigma = []
                n_deg = n-1
                for i in range(0, n):
                    sigma = math.factorial(n_deg) / ( math.factorial(i) * math.factorial(n_deg - i) )
                    Sigma.append(sigma)

                for i in range(0, n):
                    for j in range(0, n):
                        if (i-j < 0):
                            M[i,j] = 0
                        else:
                            M[i,j] = (-1)**(i-j) * Sigma[i] * math.factorial(i) / ( math.factorial(j) * math.factorial(i-j) )

                return M

            def blend(self, M, n, Theta):
                """Calculate blending function

                Arguments:
                    M {[type]} -- [spline matrix]
                    n {[type]} -- [degree of bez curve]
                    Theta {[list]} -- [list between of values between 0 and 1.]
                """                
                l = np.empty(shape=[0, n])
                ll = np.empty(shape=[0, n])
                lll = np.empty(shape=[0, n])
                llll = np.empty(shape=[0, n])

                for theta in Theta:
                    poly =       np.array([1, theta, theta**2, theta**3,    theta**4,    theta**5,     theta**6,     theta**7])
                    dot_poly =   np.array([0, 1, 2*theta, 3*theta**2,  4*theta**3,  5*theta**4,   6*theta**5,   7*theta**6])
                    ddot_poly =  np.array([0, 0, 2,   6*theta,    12*theta**2,    20*theta**3,   30*theta**4,    42*theta**5])
                    dddot_poly = np.array([0, 0, 0,  6,  24*theta,   60*theta**2,    120*theta**3,   210*theta**4])
                    
                    l = np.vstack([l, poly])
                    ll = np.vstack([ll, dot_poly])
                    lll = np.vstack([lll, ddot_poly])
                    llll = np.vstack([llll, dddot_poly])

                self.B_blend = l.dot(M)
                self.dot_B_blend = ll.dot(M)
                self.ddot_B_blend = lll.dot(M)
                self.dddot_B_blend = llll.dot(M)

    class PlotOutput:
        """Plotting class for path generator.
        """        
        def __init__(self, WP_current, WP_next, CP_opt, zeta, pg_output, i, s):
            self.WP_current = WP_current
            self.WP_next = WP_next
            self.CP_opt = CP_opt
            self.zeta = zeta
            self.pg_output = pg_output
            self.i = i
            self.s = s
          
    ### Callbacks ###

    def parameter_callback(self, config, level):
        """ Callback function for updating parameters.

        Parameters
        ----------
        config : ParameterGenerator()
            configuration parameters

        """
        self.h        = config.bez_stepsize_h
        self.zeta     = config.corridor_width_zeta
        self.u_d      = config.desired_speed_u_d
        self.mu       = config.update_law_mu
        self.epsilon[0] = self.zeta/config.epsilon_1
        self.epsilon[1] = self.zeta/config.epsilon_2
        self.epsilon[2] = self.zeta/config.epsilon_3

        return config

    def wp_listener_cme(self, path):
        """Listner for list of wps coming from CME.

        Arguments:
            path {[ros msg]} -- [List of wps]
        """
        WP_list = np.zeros((len(path.poses), 2))

        for i in range(len(path.poses)):
            wp = []
            x = path.poses[i].pose.position.x
            y = path.poses[i].pose.position.y
            wp.append(x)
            wp.append(y)
            WP_list[i] = wp

        # If new waypoints. Else do nothing.
        if not np.array_equal(WP_list, self.WP):
            #Update WP-list
            self.WP = np.copy(WP_list)
            
            # Rotate to start straight.
            R = self.rot_matrix(self.eta[2])
            for row in WP_list:
                x_rot = R[0][0] * row[0] + R[0][1] * row[1]
                y_rot = R[1][0] * row[0] + R[1][1] * row[1] 
                row[0] = x_rot
                row[1] = y_rot
            
            # Add current position.
            for row in WP_list:
                row[0] += self.eta[0]
                row[1] += self.eta[1]
            
            # run path generation
            self.path_generator_cme(WP_list, self.zeta)

    def wp_listener(self, path):
        """Listener for wps coming from local planner. NOTE: position.z is used for heading.

        Arguments:
            path {[ros msg]} -- [current and next wp.
        """

        wp_current = np.array([path.poses[0].pose.position.x, path.poses[0].pose.position.y])
        psi_current = path.poses[0].pose.position.z
        wp_next = np.array([path.poses[1].pose.position.x, path.poses[1].pose.position.y])
        psi_next =  path.poses[1].pose.position.z
        
        self.path_generator( wp_current, psi_current, wp_next, psi_next, self.zeta, self.i)
        self.i += 1

    def eta_listener(self, eta):
        """
        Listens to eta (NED) position.
        """
        deg2rad = math.pi/180.0
        # [x, y, psi]
        self.eta = np.array([eta.linear.x, eta.linear.y, deg2rad*eta.angular.z])
  
if __name__ == '__main__':
    p = GuidanceSystem()
    try:
        p.guidance_law()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass



