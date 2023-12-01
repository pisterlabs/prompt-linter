"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from vec_env import VecEnv


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


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c, l = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            # ob, ob_full, reward, done, info = env.step(data)
            ob, reward, done, info = env.step(data)
            if all(done):
                ob = env.reset()
            # remote.send((ob, ob_full, reward, done, info))
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            # ob, ob_full = env.reset()
            # remote.send((ob, ob_full))
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))

        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        # obs, obs_full, rews, dones, infos = zip(*results)
        # return np.stack(obs), np.stack(obs_full), np.stack(rews), np.stack(dones), infos
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

    def render(self, mode='human'):
        # code doesn't work all that well
        # TODO: need to clean up
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        bigimg = imgs[0]
        # bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError



# class DummyVecEnv(VecEnv):
#     def __init__(self, env_fns):
#         self.envs = [fn() for fn in env_fns]
#         env = self.envs[0]
#         VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
#         if all([hasattr(a, 'adversary') for a in env.agents]):
#             self.agent_types = ['adversary' if a.adversary else 'agent' for a in
#                                 env.agents]
#         else:
#             self.agent_types = ['agent' for _ in env.agents]
#         self.ts = np.zeros(len(self.envs), dtype='int')
#         self.actions = None
#
#     def step_async(self, actions):
#         self.actions = actions
#
#     def step_wait(self):
#         results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
#         obs, obs_full, rews, dones, infos = map(np.array, zip(*results))
#         self.ts += 1
#         for (i, done) in enumerate(dones):
#             if all(done):
#                 obs[i] = self.envs[i].reset()
#                 self.ts[i] = 0
#         self.actions = None
#         return np.array(obs), np.array(rews), np.array(dones), infos
#
#     def reset(self):
#         results = [env.reset() for env in self.envs]
#         return np.array(results)
#
#     def close(self):
#         return
