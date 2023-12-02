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
from vector_fields import TrajectoryEllipse, spheric_geo_fence, repel

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Guided mode example")
    parser.add_argument("-ti", "--target_id", dest='target_id', default=2, type=int, help="Target aircraft ID")
    parser.add_argument("-ri", "--repel_id", dest='repel_id', default=2, type=int, help="Repellant aircraft ID")
    parser.add_argument("-bi", "--base_id", dest='base_id', default=10, type=int, help="Base aircraft ID")

    args = parser.parse_args()

    interface = None
    target_id = args.target_id
    base_id = args.base_id
    repel_id = args.repel_id

    follower_id = 10
    ex = 0 ; ey = 0
    ealpha = 0
    ea = 1.1 ; eb = 1.1
    traj = TrajectoryEllipse(np.array([ex, ey]), ealpha, ea, eb)
    vel = 0.7 #m/s
    ka = 1.6 #acceleration setpoint coeff

    p_repel = np.array([0., 0., 0.])
    err = np.array([0., 0., 0.])
    V_geo = np.array([0., 0., 0.])
    V_repel = np.array([0., 0., 0.])

    try:
        g = Guidance(interface=interface, quad_ids=[target_id, base_id, repel_id])
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

                if rc.id == base_id:
                    traj = TrajectoryEllipse(np.array([rc.X[0], rc.X[1]]), ealpha, ea, eb)
                    print('Base p',rc.X[0], rc.X[1], rc.X[2])

                if rc.id == repel_id:
                    p_repel = np.array([rc.X[0], rc.X[1], rc.X[2]])
                    print('Repel p', p_repel)

 
                if rc.id == target_id: # we've found the target
                    # print('Position : ',rc.X)
                    # print('Velocity : ',rc.V)
                    # err_p = p_des - rc.X
                    # V_des = err_p * 0.4
                    V_des = traj.get_vector_field(rc.X[0], rc.X[1])*vel

                    V_repel = repel(rc.X[0], rc.X[1], rc.X[2], x_source=p_repel[0], y_source=p_repel[1], z_source=p_repel[2], strength=4.0)

                    V_geo = spheric_geo_fence(rc.X[0], rc.X[1], rc.X[2], x_source=0., y_source=0., z_source=0., strength=-0.07)

                    V_des =  V_des + V_repel[:2] + V_geo[:2]
                    # print(type(V_des))
                    err = V_des - rc.V[:2]
                    print('Desired velocity:',la.norm(V_des),err)

                    # V_repel = repel(rc.X[0], rc.X[1], rc.X[2], x_source=p_repel[0], y_source=p_repel[1], z_source=p_repel[2], strength=4)

                    # V_geo = spheric_geo_fence(rc.X[0], rc.X[1], rc.X[2], x_source=0., y_source=0., z_source=0., strength=-0.05)

                    # print("X: %.2f; Xerr-norm: %.2f; Vdes: %.2f; Verr %.2f" %(rc.X[0], la.norm(err_p), V_des[0], err_V[0]))
                    # print(err_p)
                    # print(p_des)
                    # print(err_V)

            # g.accelerate(north = err_V[0]*0.9, east = err_V[1]*0.9, down = err_V[2]*0.9, quad_id=target_id) #3D
            # err = err_V + V_repel[:2] + V_geo[:2]
            # err = V_geo[:2]
            print('Err :', err)
            g.accelerate(north = err[0]*ka, east = err[1]*ka, down = 2.0, quad_id=target_id) #2D
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