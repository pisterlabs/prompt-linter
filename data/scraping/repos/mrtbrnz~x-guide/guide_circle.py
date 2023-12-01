#!/usr/bin/python3

from __future__ import print_function

import sys
from os import path, getenv

from math import radians
from time import sleep
import numpy as np
from numpy import linalg as la
import queue

import time

# Paparazzi guidance api
from guidance_common import Rotorcraft , Guidance
from trajectory_vector_fields import traj_ellipse

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Guided mode example")
    parser.add_argument("-ti", "--target_id", dest='target_id', default=0, type=int, help="Target aircraft ID")

    args = parser.parse_args()

    interface = None
    target_id = args.target_id
    follower_id = 10
    ex = 0 ; ey = 0
    ealpha = 0
    ea = 1.1 ; eb = 1.1
    traj = traj_ellipse(np.array([ex, ey]), ealpha, ea, eb)
    vel = 2.0 #m/s
    ka = 1.6 #acceleration setpoint coeff

    try:
        g = Guidance(interface=interface, quad_ids=[target_id, follower_id])
        g.step = 0.1
        sleep(0.1)
        # g.set_guided_mode()
        sleep(0.2)
        last_target_yaw = 0.0
        total_time = 0.0
        i = 0
        while True:
            sleep(g.step)
            total_time = total_time + g.step
            for rc in g.rotorcrafts:
                rc.timeout = rc.timeout + g.step
 
                if rc.id == target_id: # we've found the target
                    # print('Position : ',rc.X)
                    # print('Velocity : ',rc.V)
                    # err_p = p_des - rc.X
                    # V_des = err_p * 0.4
                    V_des = traj.get_vector_field(rc.X[0], rc.X[1])*vel
                    # print(type(V_des))
                    err_V = V_des - rc.V[:2]
                    print('Desired velocity:',la.norm(V_des),err_V)
                    # print("X: %.2f; Xerr-norm: %.2f; Vdes: %.2f; Verr %.2f" %(rc.X[0], la.norm(err_p), V_des[0], err_V[0]))
                    # print(err_p)
                    # print(p_des)
                    # print(err_V)

            # g.accelerate(north = err_V[0]*0.9, east = err_V[1]*0.9, down = err_V[2]*0.9, quad_id=target_id) #3D
            g.accelerate(north = err_V[0]*ka, east = err_V[1]*ka, down = 2.0, quad_id=target_id) #2D
            # if abs(err_p[0]+err_p[1]) < 0.3 : 
            #     i +=1
            #     print('Passing to the next point', i)
            #     point_not_reached = False
        exit()

    except (KeyboardInterrupt, SystemExit):
        print('Shutting down...')
        # g.set_nav_mode()
        g.shutdown()
        sleep(0.6)
        exit()


if __name__ == '__main__':
    main()

#EOF