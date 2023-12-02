"""Adapt SubprocVecEnv to save environment state. Adapted from OpenAI Baselines."""

import numpy as np
from baselines.common.vec_env import VecEnv
from dl.rl import env_state_dict, env_load_state_dict
from dl import nest


class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """

    def __init__(self, env_fns):
        """
        Arguments:
        env_fns: iterable of callables functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space,
                        env.action_space)

        self.transitions = [None for _ in range(self.num_envs)]
        self.actions = None
        self.spec = self.envs[0].spec

    def step_async(self, actions):
        def _numpy_check(ac):
            if not isinstance(ac, np.ndarray):
                raise ValueError("You must pass actions as nested numpy arrays"
                                 " to DummyVecEnv.")
        nest.map_structure(_numpy_check, actions)
        self.actions = actions

    def step_wait(self):
        active = [False for _ in range(self.num_envs)]

        for e in range(self.num_envs):
            if self.transitions[e] is None or not self.transitions[e][2]:  # if episode is over:
                action = nest.map_structure(lambda ac: ac[e], self.actions)
                self.transitions[e] = self.envs[e].step(action)
                active[e] = True

        obs, rs, dones, infos = zip(*self.transitions)
        for e, info in enumerate(infos):
            info['active'] = active[e]
        obs = nest.map_structure(np.stack, nest.zip_structure(*obs))
        return obs, np.stack(rs), np.stack(dones), infos

    def reset(self, force=True):
        if not force:
            return self._reset_done_envs()
        obs = [self.envs[e].reset() for e in range(self.num_envs)]
        self.transitions = [None for _ in range(self.num_envs)]
        return nest.map_structure(np.stack, nest.zip_structure(*obs))

    def _reset_done_envs(self):
        obs = []
        for e in range(self.num_envs):
            if self.transitions[e] is None or self.transitions[e][2]:
                self.transitions[e] = None
                obs.append(self.envs[e].reset())
            else:
                obs.append(self.transitions[e][0])
        return nest.map_structure(np.stack, nest.zip_structure(*obs))

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def close_extras(self):
        for env in self.envs:
            env.close()
        self.closed = True

    def state_dict(self):
        env_states = []
        for e in self.envs:
            env_states.append(env_state_dict(e))
        return {'env_states': env_states}

    def load_state_dict(self, state_dict):
        for e, state in zip(self.envs, state_dict['env_states']):
            if isinstance(state, list):
                # this could happen if the state was saved with a subproc env
                state = state[0]
            env_load_state_dict(e, state)


if __name__ == "__main__":
    import unittest
    import gym
    from gym import Wrapper

    class StateWrapper(Wrapper):
        # hack to save random state from lunar lander env.
        def __init__(self, env):
            super().__init__(env)

        def step(self, action):
            return self.env.step(action)

        def state_dict(self):
            return {'rng': self.env.np_random.get_state()}

        def load_state_dict(self, state_dict):
            self.env.np_random.set_state(state_dict['rng'])

    def make_env(nenv, seed=0):
        def _env(rank):
            def _thunk():
                env = gym.make('LunarLander-v2')
                env = StateWrapper(env)
                env.seed(seed + rank)
                return env
            return _thunk
        return DummyVecEnv([_env(i) for i in range(nenv)])

    class TestDummyVecEnv(unittest.TestCase):
        """Test DummyVecEnv"""

        def test(self):
            nenv = 4
            env = make_env(nenv)
            obs = env.reset()
            env2 = make_env(nenv)
            obs2 = env2.reset()
            env3 = make_env(nenv, seed=1)
            obs3 = env3.reset()

            assert np.allclose(obs, obs2)
            assert not np.allclose(obs, obs3)

            for _ in range(100):
                actions = np.array([env.action_space.sample()
                                    for _ in range(nenv)])
                ob, r, done, _ = env.step(actions)
                ob2, r2, done2, _ = env2.step(actions)
                assert np.allclose(ob, ob2)
                assert np.allclose(r, r2)
                assert np.allclose(done, done2)

            env3.load_state_dict(env.state_dict())
            ob = env.reset()
            ob3 = env3.reset()
            assert np.allclose(ob, ob3)

            for _ in range(100):
                actions = np.array([env.action_space.sample()
                                    for _ in range(nenv)])
                ob, r, done, _ = env.step(actions)
                ob3, r3, done3, _ = env3.step(actions)
                assert np.allclose(ob, ob3)
                assert np.allclose(r, r3)
                assert np.allclose(done, done3)

            dones = [False for _ in range(nenv)]
            obs = [None for _ in range(nenv)]
            while not np.all(dones):
                actions = np.array([env.action_space.sample()
                                    for _ in range(nenv)])
                ob, r, new_dones, _ = env.step(actions)
                for e, d in enumerate(new_dones):
                    if dones[e]:
                        assert d
                        assert np.allclose(ob[e], obs[e])
                    obs[e] = ob[e]
                dones = new_dones
            env.reset(force=False)

    unittest.main()
