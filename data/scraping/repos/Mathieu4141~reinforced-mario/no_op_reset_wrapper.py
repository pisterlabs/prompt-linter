"""
This implementation is inspired from OpenAI
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

import gym


class NoopResetEnv(gym.Wrapper):
    NOOP_ACTION: int = 0

    def __init__(self, env, noops=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noops = noops
        assert env.unwrapped.get_action_meanings()[self.NOOP_ACTION] == "NOOP"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs = None
        for _ in range(self.noops):
            obs, _, done, _ = self.env.step(self.NOOP_ACTION)
            if done:
                obs = self.env.reset(**kwargs)
        return obs
