#!/usr/bin/env python

from __future__ import print_function

import sys
from os import path, getenv

from math import radians
from time import sleep
import numpy as np
import queue

# Deep guidance stuff
# import tensorflow as tf
import time

# from settings import Settings
# from build_neural_networks import BuildActorNetwork

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
    
    p_des = [2.0 , 0.0, 3.0]
    try:
        g = Guidance(interface=interface, quad_ids=[target_id, follower_id])
        g.step = 0.1
        sleep(0.1)
        # g.set_guided_mode()
        sleep(0.2)
        last_target_yaw = 0.0
        total_time = 0.0
            
        while True:
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
                	print("X: %.2f; Xerr: %.2f; Vdes: %.2f; Verr %.2f" %(rc.X[0], err_p[0], V_des[0], err_V[0]))
              

            # current_velocity = policy_input[7:10]                

            # Send velocity/acceleration command to aircraft!
            #g.move_at_ned_vel( yaw=-deep_guidance[0])
            #g.accelerate(north = deep_guidance[0], east = -deep_guidance[1], down = -deep_guidance[2])
            g.accelerate(north = err_V[0]*0.9, east = err_V[1]*0.9, down = err_V[2]*0.9, quad_id=target_id)
            #print("Deep guidance command: a_x: %.2f; a_y: %.2f; a_z: %.2f" %( deep_guidance[1], deep_guidance[2], deep_guidance[3]))
            # print("Time: %.2f; X: %.2f; Vx: %.2f; Ax: %.2f" %(total_time, ))



    except (KeyboardInterrupt, SystemExit):
        print('Shutting down...')
        g.set_nav_mode()
        g.shutdown()
        sleep(0.2)
        exit()


if __name__ == '__main__':
    main()

#EOF