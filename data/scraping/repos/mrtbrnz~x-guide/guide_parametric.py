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
from vector_fields import traj_parametric, Controller

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Guided mode example")
    parser.add_argument('-ids','--quad_ids', nargs='+', help='<Required> IDs of all quads used', required=True)
    parser.add_argument("-ti", "--target_id", dest='target_id', default=0, type=int, help="Target aircraft ID")

    args = parser.parse_args()

    interface = None

    # converting this input from a list of strings to a list of ints
    all_ids = list(map(int, args.quad_ids))

    ctr = Controller(L=1e-1,beta=1e-2,k1=1e-3,k2=1e-3,k3=1e-3,ktheta=0.5,s=1.9)
    traj = traj_parametric(XYZ_off=np.array([0.,0.,2.5]), XYZ_center=np.array([1.1, 1.1, -0.6]),
                 XYZ_delta=np.array([0., np.pi/2, 0.]), XYZ_w=np.array([1,2,2]), alpha=0., controller=ctr)

    ka = 1.50 #acceleration setpoint coeff

    try:
        g = Guidance(interface=interface, quad_ids=all_ids)
        g.step = 0.1
        sleep(0.1)
        # g.set_guided_mode()
        sleep(0.2)
        last_target_yaw = 0.0
        total_time = 0.0
        i = 0; w = 250.
        if len(g.rotorcrafts)>1 : g.rotorcrafts[1].gvf_parameter = 250.
        now = time.time()
        while True:
            sleep(g.step)
            dt = time.time()-now
            now = time.time()
            total_time = total_time + g.step

            if len(g.rotorcrafts)>1 :
                g.rotorcrafts[1] += 250 - (g.rotorcrafts[1] - g.rotorcrafts[0])

            for rc in g.rotorcrafts:
                rc.timeout = rc.timeout + g.step
                ux,uy,uz,uw = traj.get_control(rc.X[0], rc.X[1], rc.X[2], rc.gvf_parameter)
                V_des = np.array([ux,uy,uz])
                rc.gvf_parameter += uw*dt
                err_V = V_des - rc.V

                # g.accelerate(north = err_V[0]*0.9, east = err_V[1]*0.9, down = err_V[2]*0.9, quad_id=target_id) #3D
                g.accelerate(north = err_V[0]*ka, east = err_V[1]*ka, down = -err_V[2]*ka, quad_id=rc.id) #2D

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