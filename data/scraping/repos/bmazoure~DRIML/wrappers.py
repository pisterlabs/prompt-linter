import gym
import gym_minigrid
from gym_minigrid.wrappers import  FullyObsWrapper,RGBImgObsWrapper
from gym.envs.registration import register
import numpy as np
try:
    from aari.wrapper import AARIWrapper
except:
    pass
from collections import deque
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

from itertools import product

from .simulations import Ising

"""
Copy-paste from OpenAI baselines
"""

class IsingWrapper(gym.Wrapper):
    def __init__(self, env, width,height):
        """
        Adds Ising noise to walls
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._reset_ising()
    
    def _reset_ising(self):
        self.ising = Ising(beta=4.)
        self.ising_dim = min(self._width,self._height)
        self.config = self.ising.reset(self.ising_dim)
        self.ising_noise = self.ising.simulate_n(1).reshape(self.ising_dim,self.ising_dim,1).copy()

    def _update_ising(self):
        self.ising_noise = self.ising.simulate_n(1).reshape(self.ising_dim,self.ising_dim,1).copy()

    def _add_noise(self,frame):
        mask = np.where(frame[:self.ising_dim,:self.ising_dim,:]==[0,0,0])
        subframe = frame[:self.ising_dim,:self.ising_dim,:].copy()
        subframe[mask[0],mask[1]] = [255,0,255]*(1+self.ising_noise[mask[0],mask[1]])/2
        frame[:self.ising_dim,:self.ising_dim,:] = subframe    
        return frame

    def reset(self):
        self._reset_ising()
        state = self.env.reset()
        state = self._add_noise(state)
        return state

    def step(self, action):
        self._update_ising()
        next_state, reward, is_done, info =  self.env.step(action)
        next_state = self._add_noise(next_state)
        import matplotlib.pyplot as plt
        plt.imshow(next_state)
        plt.show()
        return next_state,reward, is_done, info

class WarpFramePocMan(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key

        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width,num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        obs = obs.astype(np.uint8)
        
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        # obs = obs.transpose(2,0,1)
            
        return obs

class HistoryWrapper(gym.Wrapper):
    def __init__(self, env, X):
        """
        Save last X frames
        """
        super().__init__(env)
        self.X = X
        self.num_colors = self.env.observation_space.shape[-1]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.env._height, self.env._width,self.X*self.num_colors),
            dtype=np.uint8,
        )
        
        self._reset_history()
    
    def _reset_history(self):
        self.history = np.zeros(shape=(self.X*self.num_colors,self.env._height, self.env._width))

    def _add_state(self,new_state):
        self.history[:-self.num_colors] = self.history[self.num_colors:]
        self.history[-self.num_colors:] = new_state.transpose(2,0,1).copy()

    def reset(self):
        self._reset_history()
        state = self.env.reset()
        self.history[-self.num_colors:] = state.transpose(2,0,1)
        return self.history.transpose(1,2,0)

    def step(self, action):
        next_state, reward, is_done, info =  self.env.step(action)
        self._add_state(next_state)
        return self.history.transpose(1,2,0),reward, is_done, info

class DistributionShiftWrapper(gym.Wrapper):
    def __init__(self, env_maker, episodes_per_env, kwargs):
        """
        Takes 2 parameters:
        nb_envs
        episodes_per_env
        """
        self.envs = []
        self.episodes_per_env = episodes_per_env

        combs = product(*list(map(lambda x:[x] if type(x) is not list else x,kwargs.values())))

        for comb in combs:
            self.envs.append( env_maker({list(kwargs.keys())[i]:comb[i] for i in range(len(comb))}) )

        self.current_env_idx = 0
        self.current_env_eps = self.episodes_per_env

        self.env = self.envs[-1]
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.just_reset = False

    def reset(self):
        if self.current_env_eps <= 0:
            self.current_env_eps = self.episodes_per_env
            print("Switched envs %d->%d"%(self.current_env_idx,( self.current_env_idx + 1 ) % len(self.envs)))
            self.current_env_idx = ( self.current_env_idx + 1 ) % len(self.envs)
            self.just_reset = True
        self.current_env_eps -= 1
        return self.envs[self.current_env_idx].reset()

    def step(self, action):
        next_state, reward, is_done, info =  self.envs[self.current_env_idx].step(action)
        if self.just_reset:
            info = {'env_changed':True}
        else:
            info = {'env_changed':False}
        return next_state, reward, is_done, info

class MonitorEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Record episodes stats prior to EpisodicLifeEnv, etc."""
        gym.Wrapper.__init__(self, env)
        self._current_reward = None
        self._num_steps = None
        self._total_steps = None
        self._episode_rewards = []
        self._episode_lengths = []
        self._num_episodes = 0
        self._num_returned = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)

        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._num_episodes += 1

        self._current_reward = 0
        self._num_steps = 0

        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        return (obs, rew, done, info)

    def get_episode_rewards(self):
        return self._episode_rewards

    def get_episode_lengths(self):
        return self._episode_lengths

    def get_total_steps(self):
        return self._total_steps

    def next_episode_results(self):
        for i in range(self._num_returned, len(self._episode_rewards)):
            yield (self._episode_rewards[i], self._episode_lengths[i])
        self._num_returned = len(self._episode_rewards)

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset.
        For environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few fr
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2, ) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, dim):
        """Warp frames to the specified size (dim x dim)."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = dim
        self.height = dim
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

"""
Custom
"""

class GridworldPreprocess(gym.ObservationWrapper):
    def __init__(self, env, seed):
        """Custom preprocess function for gridworld observations."""
        gym.ObservationWrapper.__init__(self, env)
        env.seed(seed)


    def observation(self, frame):
        # frame = frame['image'] # gives a (N,M,3) input to the network, else is a dict
        frame = frame.astype(np.uint8)
        size = 3
        frame[-size:,-size:,0] = np.random.randint(0,255,size=(size,size))
        frame[-size:,-size:,1:] = 50
        # import matplotlib.pyplot as plt
        # plt.imshow(frame)
        # plt.show()
        return frame

class GridworldPostprocess(gym.Wrapper):
    def __init__(self, env):
        """Custom postprocess function for gridworlds.
        """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = float(reward)
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class WarpFrame3D(gym.ObservationWrapper):
    def __init__(self, env, dim):
        """Warp frames to the specified size (dim x dim x 3)."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = dim
        self.height = dim
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

def wrap_deepmind_minigrid(env, dim=84, framestack=True,seed=0):
    """Configure environment for DeepMind-style gridworlds.

    Note that we assume reward clipping is done outside the wrapper.

    Args:
        dim (int): Dimension to resize observations to (dim x dim).
        framestack (bool): Whether to framestack observations.
    """
    env = RGBImgObsWrapper(env)
    env = GridworldPreprocess(env,seed,)
    env = MonitorEnv(env)
    env = WarpFrame3D(env, dim)
    # if framestack:
    #     env = FrameStack(env, 4)
    env = GridworldPostprocess(env)
    return env

def wrap_deepmind_pixelworld(env, dim=84, framestack=True,seed=0):
    """Configure environment for DeepMind-style gridworlds.

    Note that we assume reward clipping is done outside the wrapper.

    Args:
        dim (int): Dimension to resize observations to (dim x dim).
        framestack (bool): Whether to framestack observations.
    """
    env = GridworldPreprocess(env,seed)
    env = MonitorEnv(env)
    env = WarpFrame3D(env, dim)
    # if framestack:
    #     env = FrameStack(env, 4)
    env = GridworldPostprocess(env)
    return env

def wrap_deepmind(env, dim=84, framestack=True, seed=0, clip_reward=False):
    """Configure environment for DeepMind-style Atari.

    Note that we assume reward clipping is done outside the wrapper.

    Args:
        dim (int): Dimension to resize observations to (dim x dim).
        framestack (bool): Whether to framestack observations.
    """
    env.seed(seed)
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if "NoFrameskip" in env.spec.id:
        env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, dim)
    # env = ScaledFloatFrame(env)  # TODO: use for dqn?
    if clip_reward:
        env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    if framestack:
        env = FrameStack(env, 4)
    return env

class AARIproxy(gym.Env):

    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.aari_env = AARIWrapper(self.env)
        self.action_space = self.aari_env.action_space
        self.observation_space = self.aari_env.observation_space
        self.unwrapped.ale = self.aari_env.unwrapped.ale

    def seed(self,seed=None):
        return self.aari_env.seed(seed)

    def step(self,u):
        return self.aari_env.step(u)

    def reset(self):
        return self.aari_env.reset()
