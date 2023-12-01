#!/usr/bin/env python

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Guided mode example")
    parser.add_argument("-ti", "--target_id", dest='target_id', default=0, type=int, help="Target aircraft ID")

    args = parser.parse_args()

    interface = None
    target_id = args.target_id
    follower_id = 10
    
    p_des_list = [[2.0 , 0.0, 3.0], [2.0, -2.0 , 3.0], [-1. ,2. , 3.0], [-2. ,-3. , 3.0]]

    try:
        g = Guidance(interface=interface, quad_ids=[target_id, follower_id])
        g.step = 0.1
        sleep(0.1)
        # g.set_guided_mode()
        sleep(0.2)
        last_target_yaw = 0.0
        total_time = 0.0
        i = 0
        for p_des in p_des_list:
            point_not_reached = True
            while point_not_reached:
                # TODO: make better frequency managing
                sleep(g.step)
                total_time = total_time + g.step
                # print('G IDS : ',g.ids) # debug....
                # policy_input = np.zeros(Settings.TOTAL_STATE_SIZE) # initializing policy input
                for rc in g.rotorcrafts:
                    rc.timeout = rc.timeout + g.step
     
                    if rc.id == target_id: # we've found the target
                        # print('Position : ',rc.X)
                        # print('Velocity : ',rc.V)
                        err_p = p_des - rc.X
                        V_des = err_p * 0.4
                        err_V = V_des - rc.V
                        print("X: %.2f; Xerr-norm: %.2f; Vdes: %.2f; Verr %.2f" %(rc.X[0], la.norm(err_p), V_des[0], err_V[0]))
                        # print(err_p)
                        # print(p_des)
                        # print(err_V)
 
                g.accelerate(north = err_V[0]*0.9, east = err_V[1]*0.9, down = err_V[2]*0.9, quad_id=target_id)
                if abs(err_p[0]+err_p[1]) < 0.3 : 
                    i +=1
                    print('Passing to the next point', i)
                    point_not_reached = False
                #print("Deep guidance command: a_x: %.2f; a_y: %.2f; a_z: %.2f" %( deep_guidance[1], deep_guidance[2], deep_guidance[3]))
                # print("Time: %.2f; X: %.2f; Vx: %.2f; Ax: %.2f" %(total_time, ))
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