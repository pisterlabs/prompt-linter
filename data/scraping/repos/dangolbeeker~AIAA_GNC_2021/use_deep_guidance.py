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
    parser.add_argument("-f", "--filename", dest='log_filename', default='log_velocity_000', type=str, help="Log file name")
    args = parser.parse_args()

    interface = None
    target_id = args.target_id
    follower_id = args.follower_id
    log_filename = args.log_filename
    max_duration = 100000
    log_placeholder = np.zeros((max_duration, 30))
    i=0 # for log increment
    
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

    ### End deep guidance initialization stuff

        try:
            start_time = time.time()
            g = Guidance(interface=interface, quad_ids=[target_id, follower_id])
            sleep(0.1)
            g.set_guided_mode(quad_id = follower_id)
            sleep(0.2)
            last_target_yaw = 0.0
            
            if Settings.AUGMENT_STATE_WITH_ACTION_LENGTH > 0:                    
                # Create state-augmentation queue (holds previous actions)
                past_actions = queue.Queue(maxsize = Settings.AUGMENT_STATE_WITH_ACTION_LENGTH)
        
                # Fill it with zeros to start
                for i in range(Settings.AUGMENT_STATE_WITH_ACTION_LENGTH):
                    past_actions.put(np.zeros(Settings.ACTION_SIZE), False)
                    
            while True:
                # TODO: make better frequency managing
                sleep(timestep)
                # print('G IDS : ',g.ids) # debug....
                policy_input = np.zeros(Settings.TOTAL_STATE_SIZE) # initializing policy input
                for rc in g.rotorcrafts:
                    rc.timeout = rc.timeout + timestep
                    # print('rc.id',rc.id)
                    #print('rc.W',rc.W)  # example to see the positions, or you can get the velocities as well...
                    if rc.id == target_id: # we've found the target
                        policy_input[3] =  rc.X[0] # target X [north] =   North
                        policy_input[4] = -rc.X[1] # targey Y [west]  = - East
                        policy_input[5] =  rc.X[2] # target Z [up]    =   Up
                        policy_input[6] =  np.unwrap([last_target_yaw, -rc.W[2]])[1] # target yaw  [counter-clockwise] = -yaw [clockwise]
                        last_target_yaw = policy_input[6]
                        # Note: rc.X returns position; rc.V returns velocity; rc.W returns attitude
                    if rc.id == follower_id: # we've found the chaser (follower)
                        policy_input[0] =  rc.X[0] # chaser X [north] =   North
                        policy_input[1] = -rc.X[1] # chaser Y [west]  = - East
                        policy_input[2] =  rc.X[2] # chaser Z [up]    =   Up

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
                # deep guidance = [chaser_x_velocity [north], chaser_y_velocity [west], chaser_z_velocity [up], chaser_angular_velocity [counter-clockwise looking down from above]]
        
                # Adding the action taken to the past_action log
                if Settings.AUGMENT_STATE_WITH_ACTION_LENGTH > 0:
                    past_actions.put(deep_guidance)
                
                # Send velocity command to aircraft!
                g.move_at_ned_vel(north = deep_guidance[0], east = -deep_guidance[1], down = 0, quad_id = follower_id)
                print("Policy input: ", policy_input, "Deep guidance command: ", deep_guidance)
                
                # Log all input and outputs:
                t = time.time()-start_time
                log_placeholder[i,0] = t
                log_placeholder[i,1:3] = deep_guidance
                log_placeholder[i,3:5] = deep_guidance
                # log_placeholder[i,5:8] = deep_guidance_xf, deep_guidance_yf, deep_guidance_zf
                log_placeholder[i,5:5+len(policy_input)] = policy_input
                i += 1
    


        except (KeyboardInterrupt, SystemExit):
            print('Shutting down...')
            g.set_nav_mode(quad_id = follower_id)
            g.shutdown()
            sleep(0.2)
            with open(log_filename+".txt", 'wb') as f:
                np.save(f, log_placeholder[:i])
            exit()


if __name__ == '__main__':
    main()

#EOF