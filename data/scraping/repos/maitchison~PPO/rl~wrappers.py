import gym
import math
import numpy as np
import cv2
import hashlib
import collections
from gym.envs.atari import AtariEnv
from . import utils

from gym.vector import VectorEnv

from typing import Union, Optional

class EpisodicDiscounting(gym.Wrapper):
    """
    Applies discounting at the episode level
    """

    def __init__(self, env: gym.Env, discount_type, discount_gamma=1.0, discount_bias: float = 1.0):
        super().__init__(env)
        self.env = env
        self.t = 0
        self.discount_type = discount_type
        self.discount_gamma = discount_gamma
        self.discount_bias = discount_bias

    @staticmethod
    def get_discount(i: float, discount_type: str, gamma: float=1.0, discount_bias: float = 1.0):
        """
        Returns discount (gamma_i) for reward (r_i), with discounting parameter gamma.
        """
        i = i + discount_bias
        if discount_type == "finite":
            m = 1/(1-gamma)
            discount = 1.0 if i <= m else 0
        elif discount_type == "geometric":
            discount = gamma ** i
        elif discount_type == "quadratic":
            discount = 1 / (i*(i+1))
        elif discount_type == "power": # also called hyperbolic
            epsilon = 1e-1
            discount = i ** (-1-epsilon) # minus epsilon so sequence converges
        elif discount_type == "harmonic":
            discount = 1 / (i * (math.log(i)**2))
        elif discount_type == "none":
            discount = 1.0
        else:
            raise ValueError(f"Invalid discount_type {discount_type}")
        return discount

    @staticmethod
    def get_normalization_constant(k:np.ndarray, discount_type: str, gamma: float = 1.0, discount_bias: float = 1.0):
        k = k + discount_bias
        if discount_type == "finite":
            m = 1/(1-gamma)
            steps_remaining = (m-k)
            steps_remaining = np.clip(steps_remaining, 0, float('inf')) # make sure steps remaining is not negative
            normalizer = steps_remaining+1
        elif discount_type == "geometric":
            normalizer = (gamma ** k) / (1-gamma)
        elif discount_type == "quadratic":
            normalizer = 1 / k
        elif discount_type == "power": # also called hyperbolic
            epsilon = 1e-1
            normalizer = (1 / epsilon) * (k ** -epsilon)
        elif discount_type == "harmonic":
            normalizer = 1 / np.log(k)
        elif discount_type == "none":
            normalizer = 1.0
        else:
            raise ValueError(f"Invalid discount_type {discount_type}")
        return normalizer

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        assert "time" in info, "Must place timeAware wrapper before episodic discount wrapper"
        time = info["time"]
        discount = EpisodicDiscounting.get_discount(time, discount_type=self.discount_type, gamma=self.discount_gamma,
                                                    discount_bias=self.discount_bias)
        reward *= discount

        return obs, reward, done, info


class NoPassThruWrapper(gym.Wrapper):
    """
    Always returns first state after reset. Can be used to debug performance hit from running environment / wrappers.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.first = False

    def reset(self):
        self.obs = self.env.reset()
        self.first = True
        return self.obs

    def step(self, action):
        if self.first:
            self.obs, _, _, self.info = self.env.step(action)
            self.first = False
        return self.obs, 0, False, self.info


class ActionAwareWrapper(gym.Wrapper):
    """
    Includes previous on frame.
    input should be [H, W, C] of dtype np.unit8
    The action used to arrive in this state is marked onto the frame.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._process_obs(obs, -1)

    def _process_obs(self, obs, action: int):

        # input should be C, H, W, or H, W

        assert obs.dtype == np.uint8

        # draw actions we pressed on frames
        BLOCK_SIZE = 4

        if action >= 0:
            x = action * BLOCK_SIZE
            y = 0
            if len(obs.shape) == 2:
                obs[x:x+BLOCK_SIZE, y:y+BLOCK_SIZE] = 255
            else:
                C, H, W = obs.shape
                # this is a bit of a hack, procgen and atari have different channel order.
                if C < H:
                    obs[:, x:x + BLOCK_SIZE, y:y + BLOCK_SIZE] = 255
                else:
                    obs[x:x + BLOCK_SIZE, y:y + BLOCK_SIZE, :] = 255

        return obs

    def step(self, action:int):
        assert type(action) in [int, np.int, np.int32, np.int16], f"Action aware requires discrete actions, but found action of type {type(action)}"
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs, action), reward, done, info

class TimeAwareWrapper(gym.Wrapper):
    """
    Includes time on frame of last channel of observation (which is last state if using stacking)
    Observational spaces should be 2d image in format
    [..., C, H, W]
    """

    def __init__(self, env: gym.Env, log:bool=False):
        """
        Enabling log will present the log time elapsed.
        """
        super().__init__(env)
        self.log = log

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._process_obs(obs, 0)

    def _process_obs(self, obs, time_frac):
        assert obs.dtype == np.uint8
        *_, C, H, W = obs.shape

        x_point = 3 + (W-6) * time_frac

        obs[..., 0, -4:, :] = 0
        obs[..., 0, -3:-1, 3:-3] = 64
        obs[..., 0, -3:-1, 3:math.floor(x_point)] = 255
        obs[..., 0, -3:-1, math.floor(x_point)] = 64+int((x_point % 1) * (255-64))
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        assert "time_frac" in info, "Must use TimeLimitWrapper."
        if self.log:
            # log
            t = info["time"]
            if t == 0:
                max_t = 100
            else:
                max_t = info["time"] / info["time_frac"]
            time_frac = math.log(1+t) / math.log(1+max_t)
        else:
            # linear
            time_frac = np.clip(info["time_frac"], 0, 1)
        return self._process_obs(obs, time_frac), reward, done, info

class ActionHistoryWrapper(gym.Wrapper):
    """
    Includes markings on final frame in stack indicating history of actions

    [..., C, H, W]
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_history = collections.deque(maxlen=100)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.action_history.clear()
        return self._process_obs(obs)

    def _process_obs(self, obs):
        assert obs.dtype == np.uint8
        *_, C, H, W = obs.shape

        # draw history of actions at bottom final state
        n_actions = self.action_space.n
        obs[0, :n_actions, :] = 32
        for x, a in enumerate(list(self.action_history)[:W]):
            if a < 0:
                # -1 means env was ignored.
                continue
            y = a
            obs[0, y, x] = 255
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.action_history.appendleft(action)
        return self._process_obs(obs), reward, done, info

    def save_state(self, buffer):
        buffer["action_history"] = self.action_history

    def restore_state(self, buffer):
        self.action_history = buffer["action_history"]


class StateHistoryWrapper(gym.Wrapper):
    """
    Includes markings on final frame in stack indicating (compressed) history of states

    Assumes input is
    [C, H, W]
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.state_history = collections.deque(maxlen=100)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.action_history.clear()
        return self._process_obs(obs)

    def _process_obs(self, obs):
        assert obs.dtype == np.uint8
        *_, C, H, W = obs.shape

        # draw history of actions at bottom final state
        n_actions = self.action_space.n
        # we leave space for n_actions...
        obs[0, n_actions:n_actions + 49, :] = 0
        for x, state in enumerate(list(self.state_history)[:W]):
            obs[0, n_actions:n_actions+49, x] = state
        return obs

    def compressed_state(self, x):
        """
        Returns the compressed version of the state
        Input should be [C,H,W]
        Output will be [49]
        """
        x = x[-1] # take most recent on stack
        x_resized = cv2.resize(x, (7, 7), interpolation=cv2.INTER_AREA)
        assert x_resized.dtype == np.uint8
        return x_resized.ravel()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.state_history.appendleft(self.compressed_state(obs))
        return self._process_obs(obs), reward, done, info

    def save_state(self, buffer):
        buffer["state_history"] = self.state_history

    def restore_state(self, buffer):
        self.state_history = buffer["state_history"]



class HashWrapper(gym.Wrapper):
    """
    Maps observation onto a random sequence of pixels.
    This is helpful for testing if the agent is simply memorizing the environment, as no generalization between
    states is possible under this observation.
    """

    def __init__(self, env, hash_size, use_time=False):
        """
        Map observation to a hash of observation.
        """
        super().__init__(env)
        self.env = env
        self.use_time = use_time
        self.hash_size = hash_size
        self.counter = 0

    def step(self, action):

        original_obs, reward, done, info = self.env.step(action)

        if self.use_time:
            state_hash = self.counter
        else:
            state_hash = int(hashlib.sha256(original_obs.data.tobytes()).hexdigest(), 16)

        # note: named tensor would help get this shape right...
        w, h, c = original_obs.shape

        rng = np.random.RandomState(state_hash % (2**32)) # ok... this limits us to 32bits.. might be a better way to do this?

        # seed the random generator and create an random 42x42 observation.
        # note: I'm not sure how many bits the numpy random generate will use, it's posiable it's a lot less than
        # 1024. One option is then to break up the observation into parts. Another would be to just assume that the
        # number of reachable states is much much less than this, and that the chance of a collision (alaising) is
        # very low.
        new_obs = rng.randint(0, 1+1, (self.hash_size,self.hash_size), dtype=np.uint8) * 255
        new_obs = cv2.resize(new_obs, (h, w), interpolation=cv2.INTER_NEAREST)
        new_obs = new_obs[:, :, np.newaxis]

        new_obs = np.concatenate([new_obs]*c, axis=2)

        self.counter += 1

        return new_obs, reward, done, info

    def reset(self):
        self.counter = 0
        return self.env.reset()

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            info['fake_done'] = True
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

class FrameSkipWrapper(gym.Wrapper):
    """
    Performs frame skipping with max over last two frames.
    From https://github.com/openai/baselines/blob/7c520852d9cf4eaaad326a3d548efc915dc60c10/baselines/common/atari_wrappers.py
    """
    def __init__(self, env, min_skip=4, max_skip=None, reduce_op=np.max):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        if max_skip is None:
            max_skip = min_skip
        assert env.observation_space.dtype == "uint8"
        assert min_skip >= 1
        assert max_skip >= min_skip
        # most recent raw observations
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._min_skip = min_skip
        self._max_skip = max_skip
        self._reduce_op = reduce_op
        self._t = 0

    def step(self, action):
        """Repeat action, sum reward, and max over last two observations."""
        total_reward = 0.0
        done = None
        info = {}
        skip = np.random.randint(self._min_skip, self._max_skip+1)

        for i in range(skip):
            obs, reward, done, _info = self.env.step(action)
            if i >= skip - 2:
                t = i - (skip - 2)
                self._obs_buffer[t] = obs

            # combine infos, with overwriting
            if _info is not None:
                info.update(_info)

            total_reward += reward

            if done:
                break

        # first frame will be from reset and gets an empty info, or the info from the last frame of previous
        # episode. Performing increment here means second frame seen will be tagged as t=1, which is what we want.
        self._t += 1

        if done:
            # may as well output a blank frame, as this frame will (should) not be used.
            # what will happen is env will be auto-reset and the first frame of the next game will
            # be used instead.
            reduce_frame = self._reduce_op(self._obs_buffer*0, axis=0)
            self._t = 0 # for some reason I use the info from the last state as the info for the reset observation.
            # this is due to gym not having a way to get info from a reset :(
        else:
            reduce_frame = self._reduce_op(self._obs_buffer, axis=0)

        # fix up the time step
        # normally time refers to the steps in the environment, however it's convenient to instead use number
        # of interactions with the environment. Therefore we remap the 'time' statistic to the number of interactions
        # and store the original time as time_raw.
        if 'time' in info:
            info['time_raw'] = info['time']
        info['time'] = self._t

        return reduce_frame, total_reward, done, info

    def save_state(self, buffer):
        buffer["t"] = self._t

    def restore_state(self, buffer):
        self._t = buffer["t"]

    def reset(self, **kwargs):
        self._t = 0
        return self.env.reset(**kwargs)

class ClipRewardWrapper(gym.Wrapper):
    """ Clips reward to given range"""

    def __init__(self, env: gym.Env, clip: float):
        super().__init__(env)
        self.clip = clip

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if reward > self.clip or reward < -self.clip:
            info["unclipped_reward"] = reward
            reward = np.clip(reward, -self.clip, +self.clip)

        return obs, reward, done, info


class DeferredRewardWrapper(gym.Wrapper):
    """
    All rewards are delayed until given frame. If frame is -1 then uses terminal state
    """

    def __init__(self, env: gym.Env, time_limit=-1):
        super().__init__(env)
        self.env = env
        self.t = 0
        self.episode_reward = 0
        self.time_limit = time_limit

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.t += 1

        give_rewards = (self.t == self.time_limit) or ((self.time_limit == - 1) and done)

        self.episode_reward += reward

        if give_rewards:
            new_reward = self.episode_reward
            self.episode_reward = 0
        else:
            new_reward = 0
        return obs, new_reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.t = 0
        self.episode_reward = 0
        return obs

    def save_state(self, buffer):
        buffer["t"] = self.t
        buffer["episode_reward"] = self.episode_reward

    def restore_state(self, buffer):
        self.t = buffer["t"]
        self.episode_reward = buffer["episode_reward"]


class SaveEnvStateWrapper(gym.Wrapper):
    """
    Enables saving and restoring of the environment state.
    Only support atari at the moment.
    """

    def __init__(self, env: gym.Env, determanistic:bool = True):
        super().__init__(env)
        self.determanistic = determanistic

    def save_state(self, buffer):
        assert type(self.unwrapped) == AtariEnv, "Only Atari is supported for state saving/loading"
        buffer["atari"] = self.unwrapped.clone_state(include_rng=self.determanistic)

    def restore_state(self, buffer):
        assert type(self.unwrapped) == AtariEnv, "Only Atari is supported for state saving/loading"
        assert "atari" in buffer, "No state information found for Atari."
        self.unwrapped.restore_state(buffer["atari"])


class SqrtRewardWrapper(gym.Wrapper):
    """ Clips reward to given range"""

    def __init__(self, env: gym.Env, epsilon: float = 1e-3):
        super().__init__(env)
        self.epsilon = epsilon

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        sign = -1 if reward < 0 else +1
        new_reward = sign*(math.sqrt(abs(reward)+1)-1)+self.epsilon*reward
        return obs, new_reward, done, info

class RewardCurveWrapper(gym.Wrapper):
    """
    Rewards get larger over time.
    """

    def __init__(self, env: gym.Env, scale:float):
        super().__init__(env)
        self.env = env
        self.t = 0
        self.scale=scale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.t += 1
        reward = reward * self.t * self.scale
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.t = 0
        return obs

    def save_state(self, buffer):
        buffer["t"] = self.t

    def restore_state(self, buffer):
        self.t = buffer["t"]

class NormalizeObservationsWrapper(gym.Wrapper):
    """
    Normalizes observations.
    """
    def __init__(self, env, clip, shadow_mode=False, initial_state=None):
        super().__init__(env)

        self.env = env
        self.epsilon = 1e-8
        self.clip = clip
        self.obs_rms = utils.RunningMeanStd(shape=())
        self.shadow_mode = shadow_mode
        if initial_state is not None:
            self.obs_rms.restore_state(initial_state)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs_rms.update(obs)
        self.mean = self.obs_rms.mean
        self.std = np.sqrt(self.obs_rms.var)

        info["observation_norm_state"] = self.obs_rms.save_state()

        if self.shadow_mode:
            return obs, reward, done, info
        else:
            scaled_obs = (obs - self.mean) / (self.std + self.epsilon)
            scaled_obs = np.clip(scaled_obs, -self.clip, +self.clip)
            scaled_obs = np.asarray(scaled_obs, dtype=np.float32)
            return scaled_obs, reward, done, info

    def save_state(self, buffer):
        buffer["obs_rms"] = self.obs_rms.save_state()

    def restore_state(self, buffer):
        self.obs_rms.restore_state(buffer["obs_rms"])

class RewardScaleWrapper(gym.Wrapper):

    def __init__(self, env:gym.Env, scale:float):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward*self.scale, done, info


class BigRedButtonWrapper(gym.Wrapper):
    """
    Adds 1% chance to insert a big red button into the observation space. If the agent presses any action with an
    even index the episode terminates with a penality equal to all accumulated score so far.
    """

    def __init__(self, env:gym.Env, p: float = 0.01, change_actions=False):
        """
        @param p: probability that button is inserted each frame.

        Note: this will not work well on environments with negative rewards.
        """
        super().__init__(env)
        self.p = p
        self.time_since_button_shown = None
        self.action_required = 0
        self.accumulated_reward = 0
        self.change_actions = change_actions

    def step(self, action: int):

        obs, reward, done, info = self.env.step(action)

        assert obs.shape == (84, 84, 1), "Sorry big red button is hardcoded for 84x84 resolution, single channel."
        assert obs.dtype == np.uint8, "Sorry big red button is hardcoded for uint8."

        if self.time_since_button_shown == 1:
            info['button'] = self.action_required
            # we delay a little just because the environment might be stochastic
            # actually this does not matter... because stochastic is implemented up the river, by ALE.
            if action != self.action_required:
                # blow up the world
                info['pushed_button'] = True
                return obs*0, -10000, True, info

        # draw the 'button'
        if np.random.rand() < self.p:
            self.time_since_button_shown = 0
            obs //= 3
            if self.change_actions:
                self.action_required = np.random.randint(0, self.env.action_space.n)
                x_pos = 10 + (self.action_required % 4) * 13
                y_pos = 10 + (self.action_required // 4) * 13
                obs[x_pos:x_pos+10, y_pos:y_pos+10] = 255
            else:
                self.action_required = 0
                obs[42-16:42+16, 42-16:42+16] = 255

        if self.time_since_button_shown is not None:
            self.time_since_button_shown += 1

        self.accumulated_reward += reward

        return obs, reward, done, info

    def save_state(self, buffer):
        buffer["time_since_button_shown"] = self.time_since_button_shown
        buffer["accumulated_reward"] = self.accumulated_reward
        buffer["action_required"] = self.action_required

    def restore_state(self, buffer):
        self.time_since_button_shown = buffer["time_since_button_shown"]
        self.accumulated_reward = buffer["accumulated_reward"]
        self.action_required = buffer["action_required"]

    def reset(self, **kwargs):
        self.time_since_button_shown = None
        self.accumulated_reward = 0
        return self.env.reset()



class RandomTerminationWrapper(gym.Wrapper):

    def __init__(self, env:gym.Env, p: float):
        """
        Terminates environment with per step probability p.
        This can be used to create an environment with very stochastic value functions.
        """
        super().__init__(env)
        self.p = p

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or (np.random.rand() < self.p)
        return obs, reward, done, info

class LabelEnvWrapper(gym.Wrapper):
    def __init__(self, env:gym.Env, label_name:str, label_value:str):
        super().__init__(env)
        self.label_name = label_name
        self.label_value = label_value

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info[self.label_name] = self.label_value
        return obs, reward, done, info


class ZeroObsWrapper(gym.Wrapper):
    def __init__(self, env:gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs*0, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs*0


class ReturnTracker():
    """
    Tracks returns for normalization accross a (masked) vector of environmentst
    """
    def __init__(self, num_envs: int, gamma: float):
        self.ret_rms = utils.RunningMeanStd(shape=())
        self.current_returns = np.zeros([num_envs], dtype=np.float32)
        self.gamma = gamma

    def reset(self):
        self.current_returns *= 0

    def update(self, rewards:np.ndarray, dones:np.ndarray, mask:np.ndarray):
        if sum(mask) == 0:
            return
        # the self.gamma here doesn't make sense to me as we are discounting into the future rather than from the past
        # but it is what OpenAI does...
        self.current_returns[mask] = rewards[mask] + self.gamma * self.current_returns[mask] * (1 - dones[mask])
        self.ret_rms.update(self.current_returns[mask])


class VecRepeatedActionPenalty(gym.Wrapper):

    def __init__(self, env: VectorEnv, max_repeated_actions: int, penalty: float = 1):
        super().__init__(env)
        self.max_repeated_actions = max_repeated_actions
        self.penalty = penalty
        self.prev_actions = np.zeros([env.num_envs], dtype=np.int32)
        self.duplicate_counter = np.zeros([env.num_envs], dtype=np.int32)

    def reset(self, **kwargs):
        self.prev_actions *= 0
        self.duplicate_counter *= 0
        return self.env.reset()

    def step(self, actions):

        obs, rewards, dones, infos = self.env.step(actions)

        no_action_mask = (actions >= 0) # action=-1 means we ignored that environment
        mask = (actions == self.prev_actions) * no_action_mask
        self.duplicate_counter += mask
        self.duplicate_counter *= mask

        too_many_repeated_actions = (self.duplicate_counter > self.max_repeated_actions)

        infos[0]['max_repeats'] = np.max(self.duplicate_counter)
        infos[0]['mean_repeats'] = np.mean(self.duplicate_counter)

        if np.sum(too_many_repeated_actions) > 0:
            for i, repeated_action in enumerate(too_many_repeated_actions):
                if repeated_action:
                    infos[i]['repeated_action'] = actions[i]

        self.prev_actions[:] = actions[:]

        return obs, rewards - (too_many_repeated_actions * self.penalty), dones, infos

class VecNormalizeRewardWrapper(gym.Wrapper):
    """
    Normalizes rewards such that returns are roughly unit variance.
    Vectorized version.
    Also clips rewards
    """

    def __init__(
            self,
            env: VectorEnv,
            initial_state=None,
            gamma: float = 1.0,
            clip: float = 10.0,
            scale: float = 1.0,
            returns_transform=lambda x: x,
            mode: str = "rms",
            ed_type: Optional[str] = None,
            ed_bias: float = 1.0,
            ema_horizon: float = 5e6,
    ):
        """
        Normalizes returns
        mode:
            rms uses running variance over entire history,
            ema uses ema over 5M steps.
            custom requires setting of ret_std
        """
        super().__init__(env)

        self.clip = clip
        self.epsilon = 1e-2 # was 1e-8 (1e-2 will allow 10x scaling on rewards... which probably about right.)
        self.current_returns = np.zeros([env.num_envs], dtype=np.float32)
        self.ret_rms = utils.RunningMeanStd(shape=())
        self.gamma = gamma
        self.scale = scale
        self.mode = mode
        self.returns_transform = returns_transform
        self.ed_type = ed_type
        self.ed_bias = ed_bias
        self.ret_var = 0.0
        self.ema_horizon = ema_horizon
        if initial_state is not None:
            self.ret_rms.restore_state(initial_state)

    def reset(self):
        self.current_returns *= 0
        return self.env.reset()

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)

        # note:
        # we used to do this with:
        #
        # self.current_returns = rewards + self.gamma * self.current_returns
        # self.ret_rms.update(self.current_returns)
        # self.current_returns = self.current_returns * (1-dones)
        #
        # which I think is more correct, but is quite inconsistent when rewards are at terminal states.
        # I also think this matches OpenAI right?
        # now instead we do it the older way, which I think was OpenAI's older method.
        # Note: the important change here is on what happens on a transition that both gives reward and terminates.

        # ok, so follow up
        # baselines https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/vec_env/vec_normalize.py#L4
        #   they do this the v2 way, which is zero returns after update, I can show this is wrong.
        # baselines3
        #   they just ignore terminals... interesting...
        # I think my way is correct. It correlates well with the true return

        # the self.gamma here doesn't make sense to me as we are discounting into the future rather than from the past
        # but it is what OpenAI does...
        self.current_returns = rewards + self.gamma * self.current_returns * (1-dones)

        # episodic discounting return normalization
        if self.ed_type is not None:
            times = np.asarray([info.get('time', 0) for info in infos]) # during warmup we occasionally get some empty infos
            norms = EpisodicDiscounting.get_normalization_constant(times, self.ed_type, discount_bias=self.ed_bias)
        else:
            norms = 1

        self.ret_rms.update(self.returns_transform(self.current_returns/norms)) # stub /norms

        if self.mode == "ema":
            # note: we move EMA a bit faster at the beginning
            alpha = 1 - (len(dones) / min(self.ret_rms.count, self.ema_horizon))
            self.ret_var = alpha * self.ret_var + (1 - alpha) * np.var(self.current_returns)

        scaled_rewards = rewards / self.std
        # print(self.current_returns.max())
        # print(scaled_rewards.max())
        if self.clip is not None and self.clip >= 0:
            rewards_copy = scaled_rewards.copy()
            scaled_rewards = np.clip(scaled_rewards, -self.clip, +self.clip)
            clips = np.sum(rewards_copy != scaled_rewards)
            if clips > 0:
                # log if clipping occurred.
                infos[0]["reward_clips"] = clips

        scaled_rewards *= self.scale

        return obs, scaled_rewards, dones, infos

    @property
    def mean(self):
        return self.ret_rms.mean

    @property
    def std(self):
        if self.mode == "rms":
            return math.sqrt(self.ret_rms.var + self.epsilon)
        elif self.mode in ["ema", "custom"]:
            return math.sqrt(self.ret_var + self.epsilon)
        else:
            raise ValueError(f"Invalid mode {self.mode}")

    def save_state(self, buffer):
        buffer["ret_rms"] = self.ret_rms.save_state()
        buffer["ret_var"] = self.ret_var
        buffer["current_returns"] = self.current_returns

    def restore_state(self, buffer):
        self.ret_var = buffer["ret_var"]
        self.ret_rms.restore_state(buffer["ret_rms"])
        self.current_returns = buffer["current_returns"]



class MultiEnvVecNormalizeRewardWrapper(gym.Wrapper):
    """
    Normalizes rewards such that returns are unit normal.
    Supports normalization for multiple environment types.
    Vectorized version.
    Also clips rewards
    """

    def __init__(
            self,
            env: VectorEnv,
            gamma: float = 1.0,
            clip: float = 10.0,
            scale: float = 1.0,
    ):
        """
        Normalizes returns
        """
        super().__init__(env)

        self.clip = clip
        self.epsilon = 1e-8
        self.current_returns = np.zeros([env.num_envs], dtype=np.float32)
        self.normalizers = {'default': ReturnTracker(env.num_envs, gamma)}
        self.gamma = gamma
        self.scale = scale


    def reset(self):
        for k, v in self.normalizers.items():
            v.reset()
        return self.env.reset()

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)

        env_ids = []
        for info in infos:
            env_ids.append(info.get("env_id", "default"))

        scaled_rewards = rewards.copy()

        # multi-env support
        for env_id in set(env_ids):
            if env_id not in self.normalizers:
                self.normalizers[env_id] = ReturnTracker(self.env.num_envs, self.gamma)
            mask = [id == env_id for id in env_ids]

            self.normalizers[env_id].update(rewards, dones, mask)
            scaled_rewards[mask] /= math.sqrt(self.normalizers[env_id].ret_rms.var + self.epsilon)

        # clip rewards, and monitor for clipping
        if self.clip is not None:
            rewards_copy = scaled_rewards.copy()
            scaled_rewards = np.clip(scaled_rewards, -self.clip, +self.clip)
            clips = np.sum(rewards_copy != scaled_rewards)
            if clips > 0:
                # log if clipping occurred.
                infos[0]["reward_clips"] = clips

        scaled_rewards *= self.scale

        return obs, scaled_rewards, dones, infos

    @property
    def mean(self):
        return self.normalizers["default"].ret_rms.mean

    @property
    def std(self):
        return math.sqrt(self.normalizers["default"].ret_rms.var + self.epsilon)

    def save_state(self, buffer):
        buffer["normalizers"] = self.normalizers

    def restore_state(self, buffer):
        self.normalizers = buffer["normalizers"]


class VecNormalizeObservationsWrapper(gym.Wrapper):
    """
    Normalizes observations.
    Vectorized Version
    Preserves type
    """
    def __init__(self, env: VectorEnv, clip=3.0, initial_state=None, scale_mode="normal", stacked=False):
        """
        shadow_mode: Record mean and std of obs, but do not apply normalization.
        scale_mode:
            unit_normal: Observations will be float32 unit normal,
            scaled: Observations will be 0..1 scaled to uint8 where 0 = -clip, 127=0, and 255 = +clip.
            shadow: No normalization, used to monitor mu and std.
        stacked:
            if true causes normalization to be per frame rather than per stack
        """
        super().__init__(env)

        assert scale_mode in ["unit_normal", "scaled", "shadow"]

        self.env = env
        self.epsilon = 1e-4
        self.clip = clip
        self.obs_rms = utils.RunningMeanStd()
        self.scale_mode = scale_mode
        self.stacked = stacked
        if initial_state is not None:
            self.obs_rms.restore_state(initial_state)

    def step(self, action):
        """
        Input should be [B, *obs_shape] of not stacked, otherwise [B, [stack_size], *obs_shape]
        """

        obs: np.ndarray
        reward: np.ndarray

        obs, reward, done, info = self.env.step(action)
        if self.stacked:
            B, stack_size, *obs_shape = obs.shape
            self.obs_rms.update(obs.reshape(B*stack_size, *obs_shape))
        else:
            self.obs_rms.update(obs)
        self.mean = self.obs_rms.mean.astype(np.float32)
        self.std = np.sqrt(self.obs_rms.var).astype(np.float32)

        if self.scale_mode == "shadow":
            return obs, reward, done, info
        elif self.scale_mode == "unit_normal":
            scaled_obs = (obs.astype(np.float32) - self.mean) / (self.std + self.epsilon)
            scaled_obs = np.clip(scaled_obs, -self.clip, +self.clip)
            return scaled_obs, reward, done, info
        elif self.scale_mode == "scaled":
            scaled_obs = (obs.astype(np.float32) - self.mean) / (self.std + self.epsilon)
            scaled_obs = (np.clip(scaled_obs, -self.clip, +self.clip) / (self.clip*2) + 0.5) * 255
            scaled_obs = scaled_obs.astype(np.uint8)
            return scaled_obs, reward, done, info
        else:
            raise ValueError(f"Invalid scale_mode {self.scale_mode}")

    def save_state(self, buffer):
        buffer["obs_rms"] = self.obs_rms.save_state()

    def restore_state(self, buffer):
        self.obs_rms.restore_state(buffer["obs_rms"])


class MonitorWrapper(gym.Wrapper):
    """
    Records a copy of the current observation and reward into info.
    This can be helpful to retain an unmodified copy of the input.
    """

    def __init__(self, env: gym.Env, monitor_video=False):
        super().__init__(env)
        self.monitor_video = monitor_video

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.monitor_video:
            info["monitor_obs"] = obs.copy()
        info["raw_reward"] = reward
        return obs, reward, done, info

class FrameCropWrapper(gym.Wrapper):
    """
    Crops input frame.
    """

    def __init__(self, env: gym.Env, x1, x2, y1, y2):
        super().__init__(env)
        self.cropping = (slice(y1, y2, 1), slice(x1, x2, 1))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs[self.cropping]
        return obs, reward, done, info

class TimeLimitWrapper(gym.Wrapper):
    """
    From https://github.com/openai/baselines/blob/master/baselines/common/wrappers.py
    """
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        # when a done occurs we will reset and the observation returned will be the first frame of a new
        # episode, so time_frac should be 0. Remember time_frac is the time of the state we *land in* not
        # of the state we started from.
        info['time_frac'] = (self._elapsed_steps / self._max_episode_steps) if not done else 0
        info['time'] = self._elapsed_steps if not done else 0
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def save_state(self, buffer):
        buffer["_elapsed_steps"] = self._elapsed_steps

    def restore_state(self, buffer):
        self._elapsed_steps = buffer["_elapsed_steps"]


class AtariWrapper(gym.Wrapper):
    """
    Applies Atari frame warping, optional gray-scaling, and frame stacking as per nature paper.
    Note: unlike Nature the initial frame cropping is disabled by default.

    input: 210x160x3 uint8 RGB frames or 210x160 uint8 grayscale frames
    output: 84x84x1 uint8 grayscale frame (by default)

    """

    def __init__(self, env: gym.Env, width=84, height=84, interpolation=None):
        """
        Stack and do other stuff...
        Input should be (210, 160, 3)
        Output of size (width, height, 3)
        """

        super().__init__(env)

        self._width, self._height = width, height

        assert env.observation_space.dtype == np.uint8, "Invalid dtype {}".format(env.observation_space.dtype)
        assert env.observation_space.shape in [(210, 160), (210, 160, 3)], "Invalid shape {}".format(env.observation_space.shape)

        if interpolation is None:
            # sort out default interpolation
            if (width, height) == (210, 160):
                interpolation = cv2.INTER_NEAREST  # this doesn't matter as no interpolation will be done.
            elif (width, height) == (105, 80):
                interpolation = cv2.INTER_LINEAR   # faster and better with a clean scaling
            else:
                interpolation = cv2.INTER_AREA     # safest option for general resizing.

        self.n_channels = 3
        self.interpolation = interpolation

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._width, self._height, self.n_channels),
            dtype=np.uint8,
        )

    def _process_frame(self, obs):

        assert len(obs.shape) in [2, 3]

        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, 2)

        width, height, channels = obs.shape

        if (width, height) != (self._width, self._height):
            obs = cv2.resize(obs, (self._height, self._width), interpolation=self.interpolation)

        if len(obs.shape) == 2:
            obs = obs[:, :, np.newaxis]

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["channels"] = ["ColorR", "ColorG", "ColorB"]
        return self._process_frame(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._process_frame(obs)


class TimeFeatureWrapper(gym.Wrapper):
    """
    Adds time as a input feature
    Input should R^D
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        assert len(self.env.observation_space.shape) == 1, f"Input should in R^D, shape was {self.env.observation_space.shape}"
        D = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(0, 255, (D+1,), dtype=self.env.observation_space.dtype)

    @staticmethod
    def _process_frame(obs: np.ndarray, time: float):
        D = obs.shape[0]
        new_obs = np.zeros((D+1,), dtype=obs.dtype)
        new_obs[:-1] = obs
        new_obs[-1] = time
        return new_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        assert 'time_frac' in info, "must include timelimit wrapper before TimeChannelWrapper"
        obs = self._process_frame(obs, info['time_frac'])
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._process_frame(obs, 0)


class TimeChannelWrapper(gym.Wrapper):
    """
    Adds time as a channel
    Input should be in HWC order
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        H, W, C = self.env.observation_space.shape
        assert C < H, f"Input should be in HWC format, not CHW, shape was {self.env.observation_space.shape}"
        self.observation_space = gym.spaces.Box(0, 255, (H, W, C+1), dtype=np.uint8)

    def _process_frame(self, obs: np.ndarray, time: float):
        assert obs.dtype == np.uint8
        H, W, C = obs.shape
        assert C < H, "Must be channels first."
        new_obs = np.zeros((H, W, C+1), dtype=np.uint8)
        new_obs[:, :, :-1] = obs
        new_obs[:, :, -1] = time * 255
        return new_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        assert 'time_frac' in info, "must include timelimit wrapper before TimeChannelWrapper"
        obs = self._process_frame(obs, info['time_frac'])
        if "channels" in info:
            info["channels"] += ["Gray"]
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._process_frame(obs, 0)

class ChannelsFirstWrapper(gym.Wrapper):
    """
    Puts observation in channels first order

    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        H, W, C = self.env.observation_space.shape
        assert C < H, f"Input should be in HWC format, not CHW, shape was {self.env.observation_space.shape}"
        self.observation_space = gym.spaces.Box(0, 255, (C, H, W), dtype=np.uint8)

    def _process_frame(self, obs):
        return obs.transpose(2, 0, 1)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process_frame(obs), reward, done, info

    def reset(self):
        return self._process_frame(self.env.reset())

class ColorTransformWrapper(gym.Wrapper):

    def __init__(self, env, color_mode: str):

        super().__init__(env)
        self.env = env
        H, W, C = self.env.observation_space.shape

        assert C < H, f"Input should be in HWC format, not CHW, shape was {self.env.observation_space.shape}"
        assert color_mode in ["bw", "rgb", "yuv", "hsv"], f'Color mode should be one of ["bw", "rgb", "yuv", "hsv"] but was {color_mode}'
        self.expected_input_shape = (H, W, C)

        if color_mode in ["bw"]:
            assert C in [1, 3]
            output_shape = (H, W, 1)
        elif color_mode in ["rgb", "yuv", "hsv"]:
            assert C == 3, f"Expecting 3 channels, found {C}"
            output_shape = (H, W, 3)
        else:
            raise ValueError("Invalid color mode.")

        self.color_mode = color_mode
        self.observation_space = gym.spaces.Box(0, 255, output_shape, dtype=np.uint8)

    def _process_frame(self, obs: np.ndarray):

        assert obs.shape == self.expected_input_shape, f"Shape missmatch, expecting {self.expected_input_shape} found {obs.shape}"

        H, W, C = obs.shape

        if C == 1:
            # this is just a black and white frame
            return obs
        elif self.color_mode == "bw":
            return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[:, :, None]
        elif self.color_mode == "yuv":
            return cv2.cvtColor(obs, cv2.COLOR_RGB2YUV)
        elif self.color_mode == "hsv":
            return cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
        elif self.color_mode == "rgb":
            return obs
        else:
            raise ValueError(f"Invalid color_mode {self.color_mode}")

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.color_mode == "bw":
            info["channels"] = ["Gray"]
        elif self.color_mode == "rgb":
            # present rgb and yuv frames as grayscale
            info["channels"] = ["ColorR", "ColorG", "ColorB"]
        elif self.color_mode == "yuv":
            # present rgb and yuv frames as grayscale
            info["channels"] = ["ColorY", "ColorU", "ColorV"]
        elif self.color_mode == "hsv":
            # present rgb and yuv frames as grayscale
            info["channels"] = ["ColorH", "ColorS", "ColorV"]

        return self._process_frame(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._process_frame(obs)


class DelayedStateDistortionWrapper(gym.Wrapper):

    """
    After 5M frames apply an negation filter.
    """

    def __init__(self, env, delay: int):
        super().__init__(env)
        self.env = env
        self.frames_seen = 0
        self.delay = delay

    def _process_frame(self, obs: np.ndarray):
        assert obs.dtype == np.uint8
        if self.frames_seen < self.delay:
            return obs
        else:
            return 255-obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames_seen += 1
        return self._process_frame(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.frames_seen += 1
        return self._process_frame(obs)

    def save_state(self, buffer):
        buffer["frames_seen"] = self.frames_seen

    def restore_state(self, buffer):
        self.frames_seen = buffer["frames_seen"]



class NullActionWrapper(gym.Wrapper):
    """
    Allows passing of a negative action to indicate not to proceed the environment forward.
    Observation, frozen, info empty, and reward will be 0, done will be false
    Child environment will not be stepped.
    Helpful for vectorized environments.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._prev_obs = None
        self._prev_info = {}

    def step(self, action:int):
        if action < 0:
            return self._prev_obs, 0, False, self._prev_info
        else:
            obs, reward, done, info = self.env.step(action)
            self._prev_obs = obs
            self._prev_info = info
            return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_obs = obs
        return obs


class EpisodeScoreWrapper(gym.Wrapper):
    """
    Records episode length and score
    """

    def __init__(self, env):
        super().__init__(env)
        self.ep_score = 0
        self.ep_length = 0

    def step(self, action:int):
        obs, reward, done, info = self.env.step(action)
        self.ep_score += reward
        self.ep_length += 1
        info["ep_score"] = self.ep_score
        info["ep_length"] = self.ep_length
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.ep_score = 0
        self.ep_length = 0
        return obs

    def save_state(self, buffer):
        buffer["ep_score"] = self.ep_score
        buffer["ep_length"] = self.ep_length

    def restore_state(self, buffer):
        self.ep_score = buffer["ep_score"]
        self.ep_length = buffer["ep_length"]

class NoopResetWrapper(gym.Wrapper):
    """
    Applies a random number of no-op actions before agent can start playing.
    From https://github.com/openai/baselines/blob/7c520852d9cf4eaaad326a3d548efc915dc60c10/baselines/common/atari_wrappers.py
    """
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        self.noop_given = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for up to noop_max steps.
            Note: this differs from openAI's implementation in that theirs  would perform at least one noop, but
            this one may sometimes perform 0. This means a noop trained agent will do well if tested on no noop.

            Actually: if we don't do at least 1 the obs will be wrong, as obs on reset is incorrect for some reason...
            one of the wrappers makes a note of this (the stacking one I think). Because of this I always noop for
            atleast one action.

        """
        obs = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
            print(f"Forcing {noops} NOOPs.")
        else:
            noops = np.random.randint(1, self.noop_max+1)

        assert noops >= 0

        self.noop_given = noops

        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        obs, reward, done, info = self.env.step(ac)
        if self.noop_given is not None:
            info['noop_start'] = self.noop_given
            self.noop_given = None
        return obs, reward, done, info

class FrameStack(gym.Wrapper):
    """ This is the original frame stacker that works by making duplicates of the frames,
        For large numbers of frames this can be quite slow.

        Input should be h,w,c order
    """

    def __init__(self, env, n_stacks=4):

        super().__init__(env)

        assert len(env.observation_space.shape) == 3, "Invalid shape {}".format(env.observation_space.shape)
        assert env.observation_space.dtype == np.uint8, "Invalid dtype {}".format(env.observation_space.dtype)

        h, w, c = env.observation_space.shape

        assert c < h, "Must have channels first."

        self.n_stacks = n_stacks
        self.original_channels = c
        self.n_channels = self.n_stacks * self.original_channels

        self.stack = collections.deque(maxlen=n_stacks)

        for i in range(n_stacks):
            self._push_obs(np.zeros((h, w, c), dtype=np.uint8))

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, self.n_channels),
            dtype=np.uint8,
        )

    def _push_obs(self, obs):
        self.stack.appendleft(obs)

    def get_obs(self):
        return np.concatenate(self.stack, axis=-1)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._push_obs(obs)
        if "channels" in info:
            info["channels"] = info["channels"] * self.n_stacks
        return self.get_obs(), reward, done, info

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_stacks):
            self._push_obs(obs)
        return self.get_obs()

    def save_state(self, buffer):
        buffer["stack"] = self.stack

    def restore_state(self, buffer):
        self.stack = buffer["stack"]


class MontezumaInfoWrapper(gym.Wrapper):
    """
    From https://github.com/openai/random-network-distillation/blob/master/atari_wrappers.py
    """
    def __init__(self, env, room_address=3):
        """
        room_address: 3 for montezuma, 1 for pitfall
        """
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = self.env.unwrapped.ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        room_id = self.get_current_room()
        self.visited_rooms.add(room_id)
        info['room_count'] = len(self.visited_rooms)
        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode'].update(visited_rooms=self.visited_rooms.copy())
        return obs, rew, done, info

    def reset(self):
        self.visited_rooms.clear()
        return self.env.reset()

class EMAFrameStack(gym.Wrapper):
    """
        Maintain EMA of previous states with different alpha values.
    """

    def __init__(self, env, n_stacks=4, gamma=2.0):

        super().__init__(env)

        assert len(env.observation_space.shape) == 3, "Invalid shape {}".format(env.observation_space.shape)
        assert env.observation_space.dtype == np.uint8, "Invalid dtype {}".format(env.observation_space.dtype)

        c,h,w = env.observation_space.shape

        assert c in [1, 3], "Invalid shape {}".format(env.observation_space.shape)

        self.n_stacks = n_stacks
        self.original_channels = c
        self.n_channels = self.n_stacks * self.original_channels
        self.gamma = gamma

        self.stack = np.zeros((self.n_channels, h, w), dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.n_channels, h, w),
            dtype=np.uint8,
        )

    def _push_obs(self, obs):
        assert self.original_channels == 1, "Stacking does not support color at the moment."
        # alpha is ema
        for i in range(self.n_stacks):
            alpha = 1/(self.gamma ** i)
            self.stack[i] = self.stack[i] * (1-alpha) + obs[:, :, 0] * alpha

    def _get_obs(self):
        return np.clip(self.stack, 0, 255).astype(np.uint8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._push_obs(obs)
        if "channels" in info:
            info["channels"] = info["channels"] * self.n_stacks
        return self._get_obs(), reward, done, info

    def reset(self):
        obs = self.env.reset()
        for i in range(self.n_stacks):
            self.stack[i] = obs[:, :, 0]
        return self._get_obs()

    def save_state(self, buffer):
        buffer["stack"] = self.stack

    def restore_state(self, buffer):
        self.stack = buffer["stack"]


class FrameStack_Lazy(gym.Wrapper):
    # taken from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    # modified for channels first.

    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = collections.deque([], maxlen=k)

        new_shape = list(env.observation_space.shape)
        new_shape[0] *= k
        new_shape = tuple(new_shape)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=env.observation_space.dtype)

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
        result = LazyFrames(list(self.frames))
        return result

class LazyFrames(object):
    # taken from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
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
            self._out = np.concatenate(self._frames, axis=0)
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

def cast_down(x: Union[str, float, int]):
    """
    Try to convert string / float into an integer, float, or string, in that order...
    """
    try:
        if int(x) == x:
            return int(x)
    except:
        pass
    try:
        if float(x) == x:
            return float(x)
    except:
        pass
    return str(x)


def get_wrapper(env, wrapper_type) -> Union[gym.Wrapper, None]:
    """
    Returns first wrapper matching type in environment, or none.
    """
    while True:
        if type(env) == wrapper_type:
            return env
        try:
            env = env.env
        except:
            return None