# This code is from openai baseline
# https://github.com/openai/baselines/tree/master/baselines/common/vec_env

import numpy as np
from multiprocessing import Process, Pipe


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_best_params':
            best_params = env.get_best_params()
            remote.send(best_params)
        elif cmd == 'step_special':
            ob, rew, done, info, action, actions_matri, predicted_action, value = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, rew, done, info, action, actions_matri, predicted_action, value))
        elif cmd == 'record_opt_experiences':
            ob, rew, done, info, params, error = env.record_opt_experiences(data[0], data[1])
            remote.send((ob, rew, done, info, params, error))
        else:
            raise NotImplementedError


class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        pass

    def get_best_params(self):
        pass

    def record_opt_experiences(self, opt, maxiter):
        self.record_opt_experiences_async(opt, maxiter)
        return self.record_opt_experiencesp_wait()

    def special_step(self, action, action_matrix, predicted_action, value):
        self.special_step_async(action, action_matrix, predicted_action, value)
        return self.special_step_wait()

    def record_opt_experiences(self, opt, maxiter):
        self.record_opt_experiences_async(opt, maxiter)
        return self.record_opt_experiences_wait()

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def special_step_async(self, action, action_matrix, predicted_action, value):
        for remote, a, a_m, p_a, v in zip(self.remotes, action, action_matrix, predicted_action, value):
            remote.send(('step_special', [a, a_m, p_a, v]))
        self.waiting = True

    def special_step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, actions, actions_matrix, predicted_actions, values = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(actions), np.stack(actions_matrix), \
               np.stack(predicted_actions), np.stack(values)

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def get_best_params(self):
        for remote in self.remotes:
            remote.send(('get_best_params', None))
        return [remote.recv() for remote in self.remotes]

    def record_opt_experiences_async(self, opt, maxiter):
        for remote in self.remotes:
            remote.send(('record_opt_experiences', [opt, maxiter]))
        self.waiting = True

    def record_opt_experiences_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, reward, done, info, params, error = zip(*results)

        return obs, reward, done, info, params, error

    def __len__(self):
        return self.nenvs