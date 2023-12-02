#!/usr/bin/env python
# -*- coding: utf-8 -*-
import DRL.DDPG as parm
import numpy as np
import datetime
import torch
import math
from torch.utils.tensorboard import SummaryWriter
import pybullet as p
from ur5_envs.gym_ur5 import gym_ur5
from ur5_envs import models
from DRL.DDPG import DDPG
from DRL.OUnoise import OrnsteinUhlenbeckActionNoise
import matplotlib.pyplot as plt

MAX_EPISODES = 8000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 100000
c1 = 2000
c2 = 200
c3 = 120
c4 = 100
# ############This noise code is copied from openai baseline
# #########OrnsteinUhlenbeckActionNoise############# Openai Code#########
# 代表fetch的位置，桌子的位置和长宽高， 障碍物的位置和长宽高
ENV = {'obs': [0.1749, 0.48, 0, 1.3, 0.75, 0.2, 0.4, 0.6, 0.2]}


# ENV = {'obs': [0.1749, 0.48, 0, 1.3, 0.75, 0.2, 0.4, 0.6, 0.2]}

def CheckCollision(env):
    for i in range(env.sim.data.ncon):  # 表示有多少个接触力
        contact = env.sim.data.contact[i]  # 接触力的数据
        # 输出当前状态下每一个接触力对应的是那两个geom  输出名字
        name = ['obstacle', 'table', 'robot0']
        can_colission = ['robot0:base_link', ]
        links_collission = ['robot0:torso_lift_link', 'robot0:head_pan_link', 'robot0:head_tilt_link',
                            'robot0:head_camera_link', 'robot0:shoulder_pan_link', 'robot0:shoulder_lift_link',
                            'robot0:upperarm_roll_link', 'robot0:elbow_flex_link', 'robot0:forearm_roll_link',
                            'robot0:wrist_flex_link', 'robot0:wrist_roll_link', 'robot0:gripper_link',
                            'robot0:r_gripper_finger_link',
                            'robot0:l_gripper_finger_link', 'obstacle', 'table', 'robot0:base_link']
        vis = False
        # print(i)
        str1 = env.sim.model.geom_id2name(contact.geom1)
        str2 = env.sim.model.geom_id2name(contact.geom2)
        # print(str1)
        # print(str2)
        # print("\n")
        for j in can_colission:
            if str1.find(j) >= 0 or str2.find(j) >= 0:
                vis = True
        if vis:
            continue
        vis = False
        for j in range(len(links_collission)):
            for k in range(len(links_collission)):
                if str1.find(links_collission[j]) >= 0 and str2.find(links_collission[k]) >= 0:
                    # vis = True
                    # print('geom1', contact.geom1, str1)
                    # print('geom2', contact.geom2, str2)
                    return True  # 不允许自碰撞
        # if vis: # 允许自碰撞
        #     continue
        for j in name:
            if str1.find(j) >= 0 or str2.find(j) >= 0:
                # print('geom1', contact.geom1, str1)
                # print('geom2', contact.geom2, str2)
                # print("\n")
                return True
            # 如果你输出的name是None的话  请检查xml文件中的geom元素是否有名字着一个属性
            # print("wrong")
    return False


def check_space(p):
    if p[0] >= 0.8 and p[0] <= 1.8 and p[1] >= 0.2 and p[1] <= 1.2 and p[2] >= 0.4 and p[2] <= 1.3:
        return True
    return False


def plot(frame_idx, rewards, acc):
    plt.subplot(121)
    plt.plot(rewards)
    plt.ylabel("Total_reward")
    plt.xlabel("Episode")

    plt.subplot(122)
    plt.plot(acc)
    plt.ylabel("Accuracy(%)")
    plt.xlabel("Episode")
    plt.show()


def huber_loss(dis, delta=0.1):
    if dis < delta:
        return 0.5 * dis * dis
    else:
        return delta * (dis - 0.5 * delta)


if __name__ == "__main__":
    # OrderedDict([('achieved_goal', Box(3,)), ('desired_goal', Box(3,)), ('observation', Box(10,))])
    # Tesnorboard
    writer = SummaryWriter(
        'runs/{}_DDPG_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "ur5_pybullet"))
    env = gym_ur5(models.get_data_path(), disp=True)

    show_debug = False
    rl = DDPG()
    # rl.load_mode()
    var = 0.3
    # noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(7))
    total_rewards = []
    step_sums = []
    acc_epi = []
    # box = np.load('data/box' + str(0) + '.npy')
    root = [0, -1.0, .0, 1.5, .0, 1.0, .0]
    rot = root.copy()
    OBS = np.array(ENV['obs'])
    box = np.array([1.28305097, 0.63599385, 0.60611947])
    acc = 0
    last_max = -1
    action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(parm.ENV_PARAMS['action']), 0.05)
    # 主循环
    for i in range(MAX_EPISODES):
        obs = env.reset(show_debug)
        # action_noise.reset()
        # obs = env.fix_target(box)
        state = obs.copy()
        st = 0
        rw = 0
        success = False
        while True:
            p.stepSimulation()
            # ------------ choose action ------------
            # now_action = np.array([s[0], s[1], s[3]]).reshape(1, -1)
            state = np.array(state)
            # noise = np.random.normal(loc=0, scale=var, size=6)

            action = rl.choose_action(state)
            noise = action_noise()
            action += noise
            collision = False
            st += 1

            # for joint_st in range(30):
            #     next_state, dis, done, _ = env.joints_step(action / 30)
            #     collision = env.check_collision()
            #     if collision:
            #         next_state, dis, done, _ = env.joints_step(-action / 30)
            #         done = False
            #         break
            # if not check_space(next_state['achieved_goal']):
            #     # print(next_state['achieved_goal'])
            #     next_state, _, done, info = env.test_step(-action / 30)
            #     break

            next_state, dis, done, info = env.joints_step(action)
            collision = env.check_collision()
            # next_state, dis, done, _ = env.joints_step(action)
            if done:
                success = True

            a = np.sqrt(np.sum(np.square(action))).copy()
            # print(collision)
            r = -c1 * huber_loss(dis) - c2 * a - c3 * int(collision)
            rw += r
            if st == MAX_EP_STEPS:
                done = True
            rl.store_transition(state, action, r, next_state, done)
            state = next_state

            # 复原
            if collision:
                env.robot.set_joints_state(env.robot.homej)
            if rl.memory_counter > MEMORY_CAPACITY:
                p.stepSimulation()
                # print("action is :", action)
                rl.learn()
                # if st%50 == 0:
                # rl.Actor_eval
                # print("action:", action)
                # print("noise", noise)
                var *= .9995
            if done:
                print("Episode {0}, Step:{1}, total reward:{2}, average reward:{3},{4}".format(i, st, rw, rw * 1.0 / st,
                                                                                  'success' if success else '----'))
                writer.add_scalar('reward/total_reward', rw, i)
                writer.add_scalar('reward/avg_reward', rw * 1.0 / st, i)
                total_rewards.append(rw)
                step_sums.append(st)
                break
        if rl.memory_counter > MEMORY_CAPACITY and i % 100 == 0:
        # if i%100 == 0:
            print("======================test mode===========================")
            acc = 0
            # test
            for j in range(100):
                obs = env.reset(show_debug)
                action_noise.reset()
                # obs = env.fix_target(box)
                state = obs.copy()
                test_reward = 0
                st = 0
                success = False
                while True:

                    p.stepSimulation()
                    state = np.array(state)

                    action = rl.choose_action(state)
                    st += 1

                    next_state, dis, done, info = env.joints_step(action)
                    collision = env.check_collision()
                    a = np.sqrt(np.sum(np.square(action))).copy()
                    # print(collision)
                    r = -c1 * huber_loss(dis) - c2 * a - c3 * int(collision)
                    test_reward += r
                    state = next_state
                    if collision:
                        done = False
                        env.robot.set_joints_state(env.robot.homej)
                    if done:
                        success = True

                    if st == MAX_EP_STEPS:
                        done = True

                    if success:
                        acc +=1

                    if done:

                        print("In test mode, Episode {0}, Step:{1}, total reward:{2}, average reward:{3},{4}".format(j, st, test_reward, test_reward * 1.0 / st,
                                                                                               'success' if success else '----'))
                        writer.add_scalar('test_mode/total_reward', test_reward, int(i/100 + 0.5)*100 + j)
                        writer.add_scalar('test_mode/avg_reward', test_reward * 1.0 / st, int(i/100 + 0.5)*100 + j)
                        break
            print("test episode {0}, the accuracy is: {1}%".format(int(i/100 + 0.5), acc))
            acc_epi.append(acc)
            writer.add_scalar('accuracy/per_100_episode', acc, int(i / 100 + 0.5))
            if acc >= last_max:
                last_max = acc
                rl.save_mode()
            # writer.add_scalar('accuracy', acc, int(i/100))
            acc = 0
    plot(MAX_EPISODES, total_rewards, acc_epi)
