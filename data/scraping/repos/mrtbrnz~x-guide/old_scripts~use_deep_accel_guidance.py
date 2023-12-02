#!/usr/bin/env python

from __future__ import print_function

import sys
from os import path, getenv

from math import radians
from time import sleep
import numpy as np
import queue

# Deep guidance stuff
import tensorflow as tf
import time

from settings import Settings
from build_neural_networks import BuildActorNetwork

# Paparazzi guidance api
from guidance_common import Rotorcraft , Guidance


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Guided mode example")
    parser.add_argument("-ti", "--target_id", dest='target_id', default=0, type=int, help="Target aircraft ID")
    parser.add_argument("-fi", "--follower_id", dest='follower_id', default=0, type=int, help="Follower aircraft ID")
    parser.add_argument("-f", "--filename", dest='log_filename', default='log_accel_000', type=str, help="Log file name")
    parser.add_argument("-d", "--deadband", dest='deadband_radius', default='0.0', type=float, help="deadband radius")
    parser.add_argument("-no_avg", "--dont_average_output", dest="dont_average_output", action="store_true")
    args = parser.parse_args()

    interface = None
    target_id = args.target_id
    follower_id = args.follower_id
    log_filename = args.log_filename
    deadband_radius = args.deadband_radius
    max_duration = 100000
    log_placeholder = np.zeros((max_duration, 30))
    i=0 # for log increment
    
    # Flag to not average the guidance output
    dont_average_output = args.dont_average_output
    if dont_average_output:
        print("\n\nDeep guidance output is NOT averaged\n\n")
    else:
        print("\n\nDeep guidance output is averaged\n\n")
    
    timestep = Settings.TIMESTEP
    
    ### Deep guidance initialization stuff
    tf.reset_default_graph()

    # Initialize Tensorflow, and load in policy
    with tf.Session() as sess:
        # Building the policy network
        state_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, Settings.OBSERVATION_SIZE], name = "state_placeholder")
        actor = BuildActorNetwork(state_placeholder, scope='learner_actor_main')
    
        # Loading in trained network weights
        print("Attempting to load in previously-trained model\n")
        saver = tf.train.Saver() # initialize the tensorflow Saver()
    
        # Try to load in policy network parameters
        try:
            ckpt = tf.train.get_checkpoint_state('../')
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("\nModel successfully loaded!\n")
    
        except (ValueError, AttributeError):
            print("No model found... quitting :(")
            raise SystemExit
    
        #######################################################################
        ### Guidance model is loaded, now get data and run it through model ###
        #######################################################################


        try:
            start_time = time.time()
            g = Guidance(interface=interface, quad_ids=[target_id, follower_id])
            sleep(0.1)
            # g.set_guided_mode()
            sleep(0.2)
            last_target_yaw = 0.0
            total_time = 0.0
            
            last_deep_guidance = np.zeros(Settings.ACTION_SIZE)
            
            if Settings.AUGMENT_STATE_WITH_ACTION_LENGTH > 0:                    
                # Create state-augmentation queue (holds previous actions)
                past_actions = queue.Queue(maxsize = Settings.AUGMENT_STATE_WITH_ACTION_LENGTH)
        
                # Fill it with zeros to start
                for i in range(Settings.AUGMENT_STATE_WITH_ACTION_LENGTH):
                    past_actions.put(np.zeros(Settings.ACTION_SIZE), False)
                
            while True:
                # TODO: make better frequency managing
                sleep(timestep)
                total_time = total_time + timestep
                # print('G IDS : ',g.ids) # debug....
                policy_input = np.zeros(Settings.TOTAL_STATE_SIZE) # initializing policy input
                for rc in g.rotorcrafts:
                    rc.timeout = rc.timeout + timestep
                    
                    
                    """ policy_input is: [chaser_x, chaser_y, chaser_z, target_x, target_y, target_z, target_theta, 
                                          chaser_x_dot, chaser_y_dot, chaser_z_dot, (optional past action data)] 
                    """
                    
                                    
                    #print('rc.W',rc.W)  # example to see the positions, or you can get the velocities as well...
                    if rc.id == target_id: # we've found the target
                        policy_input[3] =  rc.X[0] # target X [north] =   North
                        policy_input[4] = -rc.X[1] # targey Y [west]  = - East
                        policy_input[5] =  rc.X[2] # target Z [up]    =   Up
                        policy_input[6] =  np.unwrap([last_target_yaw, -rc.W[2]])[1] # target yaw  [counter-clockwise] = -yaw [clockwise]
                        last_target_yaw = policy_input[6]
                        #print("Target position: X: %.2f; Y: %.2f; Z: %.2f; Att %.2f" %(rc.X[0], -rc.X[1], rc.X[2], -rc.W[2]))
                        # Note: rc.X returns position; rc.V returns velocity; rc.W returns attitude
                    if rc.id == follower_id: # we've found the chaser (follower)
                        policy_input[0] =  rc.X[0] # chaser X [north] =   North
                        policy_input[1] = -rc.X[1] # chaser Y [west]  = - East
                        policy_input[2] =  rc.X[2] # chaser Z [up]    =   Up                        
                        
                        policy_input[7] =  rc.V[0] # chaser V_x [north] =   North
                        policy_input[8] = -rc.V[1] # chaser V_y [west]  = - East
                        policy_input[9] =  rc.V[2] # chaser V_z [up]    =   Up
                        
                        
                        #print("Time: %.2f; Chaser position: X: %.2f; Y: %.2f; Z: %.2f; Att %.2f; Vx: %.2f; Vy: %.2f; Vz: %.2f" %(rc.timeout, rc.X[0], -rc.X[1], rc.X[2], -rc.W[2], rc.V[0], -rc.V[1], rc.V[2]))
                        # Note: rc.X returns position; rc.V returns velocity; rc.W returns attitude
                    
                # Augment state with past action data if applicable
                if Settings.AUGMENT_STATE_WITH_ACTION_LENGTH > 0:                        
                    past_action_data = np.asarray(past_actions.queue).reshape([-1]) # past actions reshaped into a column
                    
                    # Remove the oldest entry from the action log queue
                    past_actions.get(False)
                    
                    # Concatenate past actions to the policy input
                    policy_input = np.concatenate([policy_input, past_action_data])
                    
                ############################################################
                ##### Received data! Process it and return the result! #####
                ############################################################
        	    # Calculating the proper policy input (deleting irrelevant states and normalizing input)
                # Normalizing
                if Settings.NORMALIZE_STATE:
                    normalized_policy_input = (policy_input - Settings.STATE_MEAN)/Settings.STATE_HALF_RANGE
                else:
                    normalized_policy_input = policy_input
        
                # Discarding irrelevant states
                normalized_policy_input = np.delete(normalized_policy_input, Settings.IRRELEVANT_STATES)
        
                # Reshaping the input
                normalized_policy_input = normalized_policy_input.reshape([-1, Settings.OBSERVATION_SIZE])
        
                # Run processed state through the policy
                deep_guidance = sess.run(actor.action_scaled, feed_dict={state_placeholder:normalized_policy_input})[0]
                # deep guidance = [ chaser_x_acceleration [north], chaser_y_acceleration [west], chaser_z_acceleration [up] ]
                
                # Adding the action taken to the past_action log
                if Settings.AUGMENT_STATE_WITH_ACTION_LENGTH > 0:
                    past_actions.put(deep_guidance)
                    
                # Limit guidance commands if velocity is too high!
                # Checking whether our velocity is too large AND the acceleration is trying to increase said velocity... in which case we set the desired_linear_acceleration to zero.
                current_velocity = policy_input[7:10]                
                deep_guidance[(np.abs(current_velocity) > Settings.VELOCITY_LIMIT) & (np.sign(deep_guidance) == np.sign(current_velocity))] = 0 
        
                # If we are in the deadband, set the acceleration to zero!
                desired_location = np.array([policy_input[3]+3*np.cos(policy_input[6]), policy_input[4]+3*np.sin(policy_input[6]), policy_input[5]])
                current_location = policy_input[0:3]
                deep_guidance[np.abs((np.abs(current_location) - np.abs(desired_location))) < deadband_radius] = 0
                
                average_deep_guidance = (last_deep_guidance + deep_guidance)/2.0
                last_deep_guidance = deep_guidance
                
                # Send velocity/acceleration command to aircraft!
                #g.accelerate(north = deep_guidance[0], east = -deep_guidance[1], down = -deep_guidance[2])
                
                if dont_average_output:
                    g.accelerate(north = deep_guidance[0], east = -deep_guidance[1], down = -deep_guidance[2], quad_id = follower_id)
                else:
                    g.accelerate(north = average_deep_guidance[0], east = -average_deep_guidance[1], down = -average_deep_guidance[2], quad_id = follower_id) # Averaged 
                
                #g.accelerate(north = average_deep_guidance[0], east = -average_deep_guidance[1], down = -average_deep_guidance[2], quad_id = follower_id)
                #g.accelerate(north = 1, east = 0.1, down = 0)
                
                print("X: %2.2f Y: %2.2f Z: %2.2f Vx: %2.2f Vy: %2.2f Vz: %2.2f Guidance_X: %.2f, Y: %.2f, Z: %.2f" %(policy_input[0], policy_input[1], policy_input[2], policy_input[7], policy_input[8], policy_input[9], average_deep_guidance[0], average_deep_guidance[1], average_deep_guidance[2]))                
                
                # Log all input and outputs:
                t = time.time()-start_time
                log_placeholder[i,0] = t
                log_placeholder[i,1:4] = deep_guidance
                log_placeholder[i,4:7] = average_deep_guidance
                # log_placeholder[i,5:8] = deep_guidance_xf, deep_guidance_yf, deep_guidance_zf
                log_placeholder[i,7:7+len(normalized_policy_input[0])] = policy_input
                i += 1
    


        except (KeyboardInterrupt, SystemExit):
            print('Shutting down...')
            g.shutdown()
            sleep(0.2)
            with open(log_filename+".txt", 'wb') as f:
                np.save(f, log_placeholder[:i])
            exit()


if __name__ == '__main__':
    main()

#EOF