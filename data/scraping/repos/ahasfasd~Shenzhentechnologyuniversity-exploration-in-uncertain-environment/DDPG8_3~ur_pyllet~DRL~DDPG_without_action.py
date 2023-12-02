#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from robotics.fetch.reach import FetchReachEnv
from DDPG import DDPG
import numpy as np

import math
import parser
import matplotlib.pyplot as plt
import matplotlib

is_ipython = 'inline' in matplotlib.get_backend()
from IPython.display import clear_output
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if is_ipython:
    from IPython import display
MAX_EPISODES = 8000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 50000
c1 = 2000
c2 = 200
c3 = 120
c4 = 50
# ############This noise code is copied from openai baseline
# #########OrnsteinUhlenbeckActionNoise############# Openai Code#########
# 代表fetch的位置，桌子的位置和长宽高， 障碍物的位置和长宽高
ENV = {'obs': [1.1251, 0.27, 0.2, 0.8, 1.2, 0.4],
       'fetch_p': [0.1749, 0.48, 0]}


# ENV = {'obs': [0.1749, 0.48, 0, 1.3, 0.75, 0.2, 0.4, 0.6, 0.2]}
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.1, theta=.0, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def CheckCollision(env):
    for i in range(env.sim.data.ncon):  # 表示有多少个接触力
        contact = env.sim.data.contact[i]  # 接触力的数据
        # 输出当前状态下每一个接触力对应的是那两个geom  输出名字
        name = ['obstacle', 'table', 'robot0']
        can_colission = ['robot0:base_link']
        links_collission = ['robot0:torso_lift_link', 'robot0:head_pan_link', 'robot0:head_tilt_link',
                            'robot0:head_camera_link', 'robot0:shoulder_pan_link', 'robot0:shoulder_lift_link',
                            'robot0:upperarm_roll_link', 'robot0:elbow_flex_link', 'robot0:forearm_roll_link',
                            'robot0:wrist_flex_link', 'robot0:wrist_roll_link', 'robot0:gripper_link',
                            'robot0:r_gripper_finger_link',
                            'robot0:l_gripper_finger_link', 'obstacle', 'table']
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
                if (j == 14 and k == 15) or (j == 15 and k == 14):
                    continue
                if str1.find(links_collission[j]) >= 0 and str2.find(links_collission[k]) >= 0:
                    # vis = True
                    # print('geom1', contact.geom1, str1)
                    # print('geom2', contact.geom2, str2)
                    return True  # 不允许自碰撞
        # if vis: # 允许自碰撞
        #     continue
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


def plot_picture(total_rewards, step_sums, distance, acc_epi):
    plt.figure(3)
    plt.clf()
    rewards_t = torch.tensor(total_rewards, dtype=torch.float)
    steps_t = torch.tensor(step_sums, dtype=torch.float)
    dis_t = torch.tensor(distance, dtype=torch.float)
    acc_t = torch.tensor(acc_epi, dtype=torch.float)
    plt.subplot(2, 2, 1)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Total_Rewards')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.subplot(2, 2, 2)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.plot(dis_t.numpy())
    # Take 100 episode averages and plot them too
    if len(dis_t) >= 100:
        means = dis_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.subplot(2, 2, 3)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.plot(steps_t.numpy())
    # Take 100 episode averages and plot them too
    if len(steps_t) >= 100:
        means = dis_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.subplot(2, 2, 4)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy(%)')
    plt.plot(acc_t.numpy())
    # Take 100 episode averages and plot them too
    if len(acc_t) >= 100:
        means = acc_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def huber_loss(dis, delta=0.1):
    if dis < delta:
        return 0.5 * dis * dis
    else:
        return delta * (dis - 0.5 * delta)


def save_file(file_name, array):
    file = open(file_name, "w")
    file.writelines(str(array))
    file.close()


def getparse():
    pass


def plot_graph(episode, rewards):
    clear_output(True)
    plt.figure(3)
    plt.clf()
    plt.title('frame %s. reward: %s' % (episode, rewards[-1]))
    plt.xlabel('Episode')
    plt.ylabel('Total_Rewards')
    plt.plot(rewards)
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def begin1(load_weight=False, train=True, model_name="Default"):
    global MAX_EP_STEPS
    env = FetchReachEnv(model_name=model_name)
    var = 0.3
    rl = DDPG()
    maxr = -9999999
    ep_rs = []
    root = [0, -1.0, .0, 1.5, .0, 1.0, .0]
    object_pos = [1.5, 0.25, 0.45, 1.5, 0.75, 0.5, 1.5, 1.25, 0.5]
    object_size = [0.05, 0.03, 0.43, 0.04, 0.04, 0.65, 0.04, 0.07, 0.71]
    for i_episode in range(1, MAX_EPISODES):
        s = env.reset()
        env.test_set_joint(root)
        box_position = s['desired_goal']
        ep_r = 0

        success = False
        for i in range(200):
            # ------------ choose action ------------
            env.render()
            joints = env.test_get_joint()
            state = np.array([])
            state = np.append(state, joints)
            state = np.append(state, box_position)
            state = np.append(state, object_pos)
            state = np.append(state, object_size)

            action = rl.choose_action(state)
            noise = np.random.normal(loc=0, scale=var, size=7)
            action += noise
            # action[2], action[-1] = 0, 0
            for i in range(10):
                next_state, _, done, info = env.test_step(action / 10)
                env.render()
                collision = CheckCollision(env=env)
                if collision:
                    next_state, _, done, info = env.test_step(-action * 2 / 10)
                if not check_space(next_state['achieved_goal']):
                    # print(next_state['achieved_goal'])
                    next_state, _, done, info = env.test_step(-action * 2 / 10)

            dis = np.sqrt(np.sum(np.square(box_position - next_state['achieved_goal']))).copy()   # L2 distance
            # dis = (np.sum(np.square(box_position - next_state['achieved_goal']))) / 7  # MSE

            a = np.sqrt(np.sum(np.square(action))).copy()
            joints = env.test_get_joint()
            next_state = np.array([])
            next_state = np.append(next_state, joints)
            next_state = np.append(next_state, box_position)
            next_state = np.append(next_state, object_pos)
            next_state = np.append(next_state, object_size)
            if dis <= 0.05 and not collision:
                success = True
            r = -c1 * huber_loss(dis) - c3 * int(collision)
            ep_r += r

            done = 1 if success or i == 199 else 0
            rl.store_transition(state, action, r, next_state, done)
            if rl.memory_counter > MEMORY_CAPACITY:
                # env.render()
                rl.learn()
                var *= .999995

            if done:
                break

            if collision:
                env.test_set_joint(root)
            state = next_state
        print('Ep: ', i_episode,
              '| Ep_r: ', round(ep_r, 2),
              "success" if success else "-------")
        ep_rs.append(ep_r)
        plot_graph(i_episode, ep_rs)
        if maxr < ep_r:
            maxr = ep_r
            rl.save_model_dict(model_name='Three_obstacle_DDQG')
    clear_output(True)
    plt.figure(3)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Total_Rewards')
    plt.plot(ep_rs)
    plt.show()
    plt.savefig('DDPG_result_without_a')
    np.save("DDPG_reward_without_a", ep_rs)


if __name__ == "__main__":
    begin1(model_name="Three_obstacle")

# import gym
# #
# if __name__ == "__main__":
#     env = FetchReachEnv()
#     # name = "obstacle02"
#     # xyz_range = env.get_object_range(name)
#     # # print(env.get_object(name))
#     # print(env.get_object_range(name))
#     while True:
#         # env.random_set_object(name, xyz_range)
#         env.render()

