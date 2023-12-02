'''
Proximial Policy Optimization (PPO-Clip ) Training Script for Aliengo Robot
Retrivied from https://github.com/nav74neet/ppo_gazebo_tf
Adapted to Openai gym
Simulation will have large amount of iteration until agents earns higher rewards consecutively 
and saves learned model accordingly
Garen Haddeler
12.11.2020

'''
#python packages
import roslib
import random
import time
import math
import matplotlib.pyplot as plt
#RL packages
import gym
import numpy as np
import tensorflow as tf
from ppo import PPOTrain,Policy_net
# Openai and ROS packages 
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
import rospy
import rospkg
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelConfiguration

ITERATION = 1000000
GAMMA = 0.95 

rospy.init_node('ppo_aliengo_train_node')
def main():
    # Initialize OpenAI_ROS ENV
    LoadYamlFileParamsTest(rospackage_name="learning_ros", rel_path_from_package_to_file="config", yaml_file_name="aliengo_stand.yaml")
    env = StartOpenAI_ROS_Environment('AliengoStand-v0')
    time.sleep(3)
    # Initialize PPO agent
    Policy = Policy_net('policy',ob_space =3,act_space = 8 )
    Old_Policy = Policy_net('old_policy',ob_space =3, act_space = 8 )
    PPO = PPOTrain(Policy, Old_Policy, gamma=GAMMA)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        obs = env.reset()
        reward = 0
        success_num = 0
        scores = []
       
        for iteration in range(ITERATION):  
            observations = []
            actions = []
            v_preds = []
            rewards = []
            while  not rospy.is_shutdown():  # until ros is not shutdown
                # prepare to feed placeholder Policy.obs
                obs = np.stack([obs]).astype(dtype=np.float32)
                act, v_pred = Policy.act(obs=obs, stochastic=True)
                #print('act: ',act, 'v_pred: ',v_pred )

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)
                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)
                #execute according to action
                next_obs, reward, done, _= env.step(act)  
                time.sleep(0.01)
                if done:
                    # next state of terminate state has 0 state value
                    v_preds_next = v_preds[1:] + [0]  
                    obs = env.reset()
                    break
                else:
                    obs = next_obs
            #scores store for visualization        
            scores.append(sum(rewards)) 
            #if consectuvely 10 times has high reward end the training
            if sum(rewards) >=400:
                success_num += 1
                print("Succes number: " + str(success_num))
                if success_num >= 5:
                    saver.save(sess, 'model_train/model_ppo.ckpt')
                    print('Clear!! Model saved.')
                if success_num >= 10:
                    saver.save(sess, 'model_train/model_ppo.ckpt')
                    print('Finished! ')
                    break
    
            else:
                success_num = 0

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, [len(observations), 3])
            actions = np.array(actions).astype(dtype=np.int32)
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
            #calculate generative advantage estimator score
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) 
            print('gaes', gaes)
            #assign current policy params to previous policy params
            PPO.assign_policy_parameters()

            inp = [observations, actions, rewards, v_preds_next, gaes]

            # PPO train
            for epoch in range(4):
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          rewards=sampled_inp[2],
                          v_preds_next=sampled_inp[3],
                          gaes=sampled_inp[4])
        plt.plot(scores)
        plt.show()
        env.stop()


if __name__ == '__main__':
    main()
