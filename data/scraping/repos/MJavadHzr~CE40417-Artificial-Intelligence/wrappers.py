from gym.core import ObservationWrapper, Wrapper, RewardWrapper
from gym.spaces import Box
import cv2
import numpy as np


# Some taken from OpenAI baselines.

class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env, gray_scale=False):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (50, 50)  # TODO: <YOUR CODE>
        self.observation_space = Box(0.0, 1.0, (
            self.img_size[0], self.img_size[1], 1 if gray_scale else env.observation_space.shape[2]))
        self.gray_scale = gray_scale

    def _to_gray_scale(self, rgb, channel_weights=[0.6, 0.3, 0.1]):
        #todo
        gray_img = np.zeros((rgb.shape[0], rgb.shape[1], 1))
        return gray_img

    def observation(self, img):
        """what happens to each observation"""
        img = img[35:475, 37:465, :].astype('float32')
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]), interpolation = cv2.INTER_AREA)
        if self.gray_scale:
            img = self._to_gray_scale(img)
        img = img/225
        # Here's what you need to do:
        #  * crop image, remove irrelevant parts
        #  * resize image to self.img_size
        #     (use imresize from any library you want,
        #      e.g. opencv, skimage, PIL, keras)
        #  * cast image to grayscale (in case of breakout)
        #  * convert image pixels to (0,1) range, float32 type

        # TODO: complete observation descaling
        processed_img = img
        return processed_img


class ClipRewardEnv(RewardWrapper):
    def __init__(self, env):
        RewardWrapper.__init__(self, env)

    def reward(self, reward):
        # TODO: you may complete this section as you please
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class MaxAndSkipEnv(Wrapper):
    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8)
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


class FireResetEnv(Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
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


class EpisodicLifeEnv(Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

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
