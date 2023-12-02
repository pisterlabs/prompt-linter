#!/usr/bin/env python
import gym
from gym import wrappers
import numpy
from stable_baselines3 import DQN, A2C
from sb3_contrib import QRDQN
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import signal
import os
import torch as th
from stable_baselines3.common.monitor import Monitor
from openai_ros import normalize, timeAware, myCallBack

# [Tri Huynh] Use saved trained
#########################################################################
model_name = "/sb3_qrdqn_s1_v1/model"
pretrained_model_name= "/sb3_qrdqn_s1_v1/model"
load = False
load_policy = False
env = None

def handler(signum, frame):
    # agent.save(model_name, outdir)
    agent.save(outdir + model_name)
    agent.save_replay_buffer(outdir + model_name+ "_buffer")
    agent.policy.save(outdir + model_name + "_policy")
    rospy.logwarn('Saved model before break this training process')
    exit(1)

signal.signal(signal.SIGINT, handler)
##########################################################################

if __name__ == '__main__':
    rospy.init_node('dqn_solver', anonymous=True, log_level=rospy.WARN)
    
    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = "TurtleBot3Navigation-v0"
    env = StartOpenAI_ROS_Environment( task_and_robot_environment_name) # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Loads parameters from the ROS param server
    lr = 0.00025
    alpha_decay= 0.01
    batch_size = 128
    gamma = .99
    rospackage_name = "my_turtlebot3_openai_example"

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    outdir = pkg_path + '/models'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        rospy.logfatal("Created folder="+str(outdir))

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[64, 64],
                     n_quantiles=50)
    max_env_steps= 200
    env._max_episode_steps = max_env_steps
    env = timeAware.MyTimeAwareObservation(env)


    log_dir = outdir + model_name + "_callback"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    callback = myCallBack.SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, verbose=1)
    
    if load:
        agent = QRDQN.load(outdir + pretrained_model_name, env=env)
        agent.load_replay_buffer(outdir + pretrained_model_name+ "_buffer")
    elif load_policy:
        oldAgent = QRDQN.load(outdir + pretrained_model_name, env=env)
        print(oldAgent.policy_kwargs)
        agent = QRDQN(
                batch_size=batch_size, 
                learning_rate= lr, 
                env=env,  
                gamma=gamma,
                tensorboard_log='/home/pronton/catkin_ws/src/RL4TurtleBot3Navigation/src/my_turtlebot3_openai_example/tensorboard/', 
                policy='MlpPolicy', 
                train_freq= (1, "step"),
                gradient_steps=1, # gradient 1 time per 1 episode
                target_update_interval=200, # each ...(200) steps, we update target_net (load weight from policy_net)
                learning_starts=64, 
                max_grad_norm = 10, # set high -> don care
                exploration_fraction=1,
                exploration_final_eps=.05,
                exploration_initial_eps=1, 
                policy_kwargs=oldAgent.policy_kwargs,
                verbose=1,
                create_eval_env=True,
                )
        del oldAgent
        agent.load_replay_buffer(outdir + pretrained_model_name+ "_buffer")
    else:
        agent = QRDQN(
                batch_size=batch_size, 
                learning_rate= lr, 
                env=env,  
                gamma=gamma,
                tensorboard_log='/home/pronton/catkin_ws/src/RL4TurtleBot3Navigation/src/my_turtlebot3_openai_example/tensorboard/', 
                policy='MlpPolicy', 
                train_freq= (1, "step"),
                gradient_steps=1, # gradient 1 time per 1 episode
                target_update_interval=200, # each ...(200) steps, we update target_net (load weight from policy_net)
                learning_starts=64, 
                max_grad_norm = 10, # set high -> don care
                exploration_fraction=1,
                exploration_final_eps=.05,
                exploration_initial_eps=1, 
                policy_kwargs=policy_kwargs,
                verbose=1,
                create_eval_env=True,
                )
    agent.learn(total_timesteps=30000, reset_num_timesteps=True, callback=callback)

    # now save the replay buffer too
    agent.save(outdir + model_name)
    agent.save_replay_buffer(outdir + model_name+ "_buffer")
    agent.policy.save(outdir + model_name + "_policy")
    env.close()