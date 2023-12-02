#!/usr/bin/env python
'''
The base envs for shadow hand. Most of them are from openai gym env.
'''
from os import path
import copy
import numpy as np
from enum import Enum

from gym import error, spaces
from gym.utils import seeding
import gym

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


class BaseEnv(gym.Env):
    """Superclass for all MuJoCo environments"""

    def __init__(self, model_path, initial_pos_dict, skip_frame):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = path.join(path.dirname(
                __file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=skip_frame)
        self.data = self.sim.data
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self.seed()

        self.set_pos_by_dict(initial_pos_dict)
        self.initial_state = copy.deepcopy(self.sim.get_state())

    @property
    def dt(self):
        return self.model.opt.timestep * self.sim.nsubsteps

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 640, 480
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def close(self):
        if self.viewer is not None:
            self.viewer.finish()
            self.viewer = None

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        obs = self._get_obs()
        return obs

    def set_state(self, qpos, qvel):
        assert qpos.shape == (
            self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def set_pos_by_dict(self, pos_dict):
        for name, value in pos_dict.items():
            self.data.set_joint_qpos(name, value)
        self.sim.forward()

    # methods to override:
    # ----------------------------

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self._viewer_setup()
        return self.viewer

    def _get_action_space(self):
        """Get the action spaces for for the envrionment.
        """
        raise NotImplementedError

    def _get_observation_space(self):
        """Get the observation spaces for for the envrionment.
        """
        raise NotImplementedError

    def _get_obs(self):
        """Get observation
        """
        raise NotImplementedError

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        raise NotImplementedError

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass
    # -----------------------------

class ControlType(Enum):
    Force = 0
    Velocity = 1
    Position = 2


class ActionRangeType(Enum):
    Raw = 0
    Normalized = 1
    RelNormalized = 2


class ShadowBaseEnv(BaseEnv):
    def __init__(self, model_path, initial_pos_dict, skip_frame, control_type=ControlType.Position):
        BaseEnv.__init__(self, model_path, initial_pos_dict, skip_frame)

        self.control_type = control_type
        if control_type != ControlType.Position:
            assert 0

    def _viewer_setup(self):
        lookat = self.data.get_body_xpos('robot0:palm')
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.5
        self.viewer.cam.azimuth = 0.
        self.viewer.cam.elevation = -45.

    def take_action(self, action, action_range_type=ActionRangeType.Normalized):
        assert action.shape == (20,)

        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.

        if action_range_type == ActionRangeType.Raw:
            self.sim.data.ctrl[:] = action
        elif action_range_type == ActionRangeType.Normalized:
            actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
            self.sim.data.ctrl[:] = actuation_center + action * actuation_range

        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])
        self.sim.step()

    def get_current_actuator_pos(self):
        pos = np.zeros(20)
        qpos = self.data.qpos.copy()
        # wrist
        pos[0:2] = qpos[0:2]
        # ff
        pos[2:4] = qpos[2:4]
        pos[4] = qpos[4] + qpos[5]
        # mf
        pos[5:7] = qpos[6:8]
        pos[7] = qpos[8] + qpos[9]
        # rf
        pos[8:10] = qpos[10:12]
        pos[10] = qpos[12] + qpos[13]
        # lf
        pos[11:14] = qpos[14:17]
        pos[14] = qpos[17] + qpos[18]
        # th
        pos[15:20] = qpos[19:24]
        return pos

class ShadowGoalBaseEnv(ShadowBaseEnv):
    """ This class is used to work with openai.baselines.her

    HER has some special requirements on the env API:
    1. observation is a dict, with keys: observation, desired_goal, achieved_goal
    2. the 'info', which is a return value from self.step(action), must have a key: 'is_success'
    3. need to implement a method: self.compute_reward(self, achieved_goal, goal, info),
       the dimension of achieved_goal and goal have two cases:
       case 1: (self.goal.shape,), e.g. (7,)
       case 2: (number of samples, self.goal.shape), e.g. (200, 7)
    """

    def __init__(self, model_path, initial_pos_dict, skip_frame):
        ShadowBaseEnv.__init__(self, model_path, initial_pos_dict, skip_frame)
        self.check_observation_space()

    def compute_reward(self, achieved_goal, goal, info):
        raise NotImplementedError

    def check_observation_space(self):
        assert('observation' in self._get_obs())
        assert('achieved_goal' in self._get_obs())
        assert('desired_goal' in self._get_obs())

