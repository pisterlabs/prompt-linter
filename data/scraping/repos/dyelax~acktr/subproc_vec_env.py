# From OpenAI Baselines

import numpy as np
from multiprocessing import Process, Pipe
from atari_wrapper import get_episodic_life_env

class VecEnv(object):
    """
    Vectorized environment base class
    """
    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        raise NotImplementedError
    def reset(self):
        """
        Reset all environments
        """
        raise NotImplementedError
    def close(self):
        pass

def worker(i, remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()

    ep_reward = 0
    num_eps = 0
    env_steps = 0

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)

            env_steps += 1
            ep_reward += reward

            info['real_done'] = False
            if done:
                ob = env.reset()

                info['real_done'] = True
                try:
                    if get_episodic_life_env(env).was_real_done_last_reset:
                        if i == 0:
                            info['ep_reward'] = ep_reward
                            info['num_eps'] = num_eps
                            info['env_steps'] = env_steps

                        num_eps += 1
                        ep_reward = 0
                    else:
                        info['real_done'] = False
                except:
                    num_eps += 1
                    pass

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
            remote.send((env.action_space, env.observation_space))
        else:
            raise NotImplementedError


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
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(i, work_remote, remote, CloudpickleWrapper(env_fn)))
            for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns))]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

        self.num_steps = 0


    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
            self.num_steps += 1
        results = [remote.recv() for remote in self.remotes]
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

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)
