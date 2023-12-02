#!/usr/bin/env python
import gym
from gym import wrappers
import numpy
from stable_baselines3 import DQN, A2C, SAC, DDPG
from sb3_contrib import QRDQN
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import signal
import os
import torch as th
from openai_ros import normalize, timeAware, myCallBack
from gym.wrappers import TimeLimit, TimeAwareObservation
from stable_baselines3.common.monitor import Monitor

# [Tri Huynh] Use saved trained 
#########################################################################
model_name = "/sb3_sac_s2_v6/model"
pretrained_model_name= "/sb3_sac_s1_v9/model"
enjoy = True
load = False
load_policy = False
env = None

def handler(signum, frame):
    if enjoy: exit(1)
    agent.save(outdir + model_name)
    agent.save_replay_buffer(outdir + model_name+ "_buffer")
    agent.policy.save(outdir + model_name + "_policy")
    rospy.logwarn('Saved model before break this training process')
    exit(1)

signal.signal(signal.SIGINT, handler)
##########################################################################

if __name__ == '__main__':
    rospy.init_node('sac_solver', anonymous=True, log_level=rospy.WARN)
    
    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = "TurtleBot3Navigation-v2"
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

    policy_kwargs = dict(activation_fn = th.nn.ReLU,
                     net_arch=[64, 64, 64]) 
    max_env_steps = 200
    env._max_episode_steps = max_env_steps
    env = timeAware.MyTimeAwareObservation(env)

    # Create log dir
    log_dir = outdir + model_name + "_callback"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    callback = myCallBack.SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, verbose=1)
    # env = normalize.NormalizeActionWrapper(env) #SAC squash output to [-1, 1] area
    if enjoy:
        agent = SAC.load(outdir + pretrained_model_name, env=env)
        obs = env.reset()
        for i in range(1000):
            action, _states = agent.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            # env.render() 
        env.close()
        exit(0)
    if load:
        agent = SAC.load(outdir + pretrained_model_name, env=env)
        agent.load_replay_buffer(outdir + pretrained_model_name+ "_buffer")
    elif load_policy:
        oldAgent = SAC.load(outdir + pretrained_model_name, env=env)
        print(oldAgent.policy_kwargs)
        agent = SAC(
                    batch_size=batch_size, 
                    learning_rate= lr, 
                    env=env,  
                    gamma=gamma,
                    tensorboard_log='/home/pronton/catkin_ws/src/RL4TurtleBot3Navigation/src/my_turtlebot3_openai_example/tensorboard/', 
                    policy="MlpPolicy",
                    train_freq= (1, "step"),
                    gradient_steps=1, # gradient 1 time per 1 episode
                    target_update_interval=200, # each ...(200) steps, we update target_net (load weight from policy_net)
                    learning_starts=batch_size,
                    policy_kwargs= oldAgent.policy_kwargs,
                    verbose=1,
                    seed=42,
                    use_sde=True,
                    sde_sample_freq=-1,
                    use_sde_at_warmup=True,
                    tau=.95, 
                    create_eval_env=True,
                    )
        # agent.set_parameters(oldAgent.get_parameters())
        del oldAgent
        agent.load_replay_buffer(outdir + pretrained_model_name+ "_buffer")
    else:
        agent = SAC(
                    batch_size=batch_size, 
                    learning_rate= lr, 
                    env=env,  
                    gamma=gamma,
                    tensorboard_log='/home/pronton/catkin_ws/src/RL4TurtleBot3Navigation/src/my_turtlebot3_openai_example/tensorboard/', 
                    policy='MlpPolicy', 
                    train_freq= (1, "step"),
                    gradient_steps=1, # gradient 1 time per 1 episode
                    target_update_interval=200, # each ...(200) steps, we update target_net (load weight from policy_net)
                    learning_starts=batch_size,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    seed=42,
                    use_sde=True,
                    sde_sample_freq=-1,
                    use_sde_at_warmup=True,
                    tau=.95, 
                    create_eval_env=True,
                    )
    agent.learn(total_timesteps=10000000, reset_num_timesteps=True, callback=callback)

    # now save the replay buffer too
    agent.save(outdir + model_name)
    agent.save_replay_buffer(outdir + model_name+ "_buffer")
    agent.policy.save(outdir + model_name + "_policy")
    env.close()
