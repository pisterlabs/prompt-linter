#!/usr/bin/env python3

import numpy as np
import sys

import math 
import time
import rospy

from guidance_talker import guidance_talker

def main():
    
    # control parameter
    control_param = {'position_control_frequency':1,'attitude_control_frequency':1}
    
    trajopt_param = {'obstacle_name': ['sc2','asteriod'],
                    'terminal_condition':np.array([0,0,0,0,0,0]),
                    'scp_param': {'error_tolerance': 0.001, 'trust': 20000, 'beta': 0.9, 'alpha' :1.2 , 'iter_max' : 10 ,'slack' : True},
                    'control_cost': 'sum_squares',
                    'control_limits' : {'u_max':0.45,'u_min':0.0},
                    'nominal_trajectory': [True]} # nominal trajectory will be construted online


    talker = guidance_talker(control_param,trajopt_param)

if __name__ == "__main__":
    # read the spacecraft name from the launch file argument 
    main()

    