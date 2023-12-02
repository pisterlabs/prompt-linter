"""
Model imported from OpenAI Gym environements:

Reacher: Mujoco environment
"""

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class FDM_ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        self.goal = [-0.11977868, 0.11620387] ############################## GOAL

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        return ob, 0, 0, 0

    def reset_model(self, state):
        qpos = np.zeros(4)
        
        # theta1 = -inf to +inf
        qpos[0] = np.arctan2(state[2], state[0])
        # theta2 = -pi to +pi
        qpos[1] = np.arctan2(state[3], state[1])
        
        qpos[-2:] = self.goal
        
        qvel = np.zeros(4)
        qvel[:2] = state[6:8]
        qvel[-2:] = 0

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

fdm_env = FDM_ReacherEnv()

def fdm_cont(state, action):
    # Set state
    fdm_env.reset_model(state)

    # valid actions are -1.0 to 1.0
    nstate, _, _, _ = fdm_env.step(action)

    # # theta1 = -inf to +inf
    # theta1 = np.arctan2(nstate[2], nstate[0])
    # # theta2 = -pi to +pi
    # theta2 = np.arctan2(nstate[3], nstate[1])

    # l1 = 0.1  # Length of arm 1
    # l2 = 0.11 # Length of arm 2

    # xpos = l1*nstate[0] + l2*np.cos(theta1+theta2)
    # ypos = l1*nstate[2] + l2*np.sin(theta1+theta2)

    return nstate

# Test script
# python
# from models.fdm_reacher import *
# import gym
# env = gym.make('Reacher-v2')
# st = env.reset()
# st1 = fdm_cont(st, [-0.8,0.8])
# st1 = fdm_cont(st1, [-0.8,0.8])
# st1 = fdm_cont(st1, [-0.8,0.8])
# st1 = fdm_cont(st1, [-0.8,0.8])
# st2 = env.step([-0.8,0.8])
# st2 = env.step([-0.8,0.8])
# st2 = env.step([-0.8,0.8])
# st2 = env.step([-0.8,0.8])
# st1
# st2