import itertools
import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union
from isaacgymenvs.tasks.base.vec_task import VecTask
from tianshou.env.venvs import BaseVectorEnv
from tianshou.env.worker import EnvWorker
# from tianshou.data.buffer.manager import ReplayBufferManager
from tianshou.data import VectorReplayBuffer
from tianshou.data import Batch
from tianshou.data.batch import _alloc_by_keys_diff, _create_value
import torch

ID_TYPE = Optional[Union[int, List[int], np.ndarray]]


class NestedEnvWorker(EnvWorker):

    def __len__(self):
        return len(self.env)

    def get_env_attr(self, key: str) -> Any:
        return [getattr(self.env, key) for _ in range(len(self))]

    def set_env_attr(self, key: str, value: Any) -> None:
        setattr(self.env, key, value)


class IGMAEnvWorker(NestedEnvWorker):
    """Dummy worker used in sequential vector environments."""

    def __init__(self, env_fn: Callable[[], VecTask]) -> None:
        ret = env_fn()
        if isinstance(ret, (list, tuple)):
            env, ind = ret
        else:
            env, ind = ret, None
        self.env = env
        self.ind = ind
        self.num_envs = self.env.num_envs if ind is None else len(list(ind))
        super().__init__(env_fn)

    def __len__(self):
        return self.num_envs

    def reset(self) -> Any:
        return self.env.reset()

    @staticmethod
    def wait(  # type: ignore
            workers: List, wait_num: int, timeout: Optional[float] = None) -> List:
        # Sequential EnvWorker objects are always ready
        return workers

    def send(self, action: Optional[Any], sid: ID_TYPE = None) -> None:
        if action is None:
            if sid is None:
                obs_dict = self.env.reset()  # type: ignore
                self.result = obs_dict['obs']
            else:
                try:
                    obs_dict = self.env.reset(indices=sid)  # type: ignore
                    self.result = obs_dict['obs']
                except Exception:
                    self.result = self.last_obs[sid]
        else:
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).to(device=self.env.device)
            obs_dict, rew_buf, reset_buf, extras = self.env.step(action)  # type: ignore
            rew_buf = rew_buf.cpu().numpy()
            reset_buf = reset_buf.cpu().numpy()
            self.last_obs = obs_dict['obs']
            self.result = self.last_obs, rew_buf, reset_buf, extras

    def seed(self, seed: Optional[int] = None) -> List[int]:
        for s in self.action_space:
            s.seed(seed)
        return getattr(self.env, 'seed', lambda x: x)(seed)

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        self.env.close()


class NestedVectorEnv(BaseVectorEnv):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        len_envs = [getattr(w, '__len__', lambda: 1)() for w in self.workers]
        num_envs = sum(len_envs)
        self.num_workers = len(self.workers)
        self.len_envs = len_envs
        self.beg_envs = [sum(len_envs[:i]) for i in range(self.num_workers)]
        self.end_envs = [self.beg_envs[i] + self.len_envs[i] for i in range(self.num_workers)]
        self.wait_num = num_envs if self.wait_num == self.env_num else self.wait_num
        self.env_num = num_envs
        # self.ready_id = list(range(self.env_num))

    def _wrap_id(self, id: ID_TYPE = None) -> Union[List[int], np.ndarray]:
        if id is None:
            return list(range(self.num_workers))
        pid = [id] if np.isscalar(id) else id  # type: ignore
        if len(pid) == pid[-1] - pid[0] + 1:
            return [i for i in range(self.num_workers) if not (self.beg_envs[i] > pid[-1] or self.end_envs[i] < pid[0])]
        else:
            return [i for i in range(self.num_workers) if any(self.beg_envs[i] <= j < self.end_envs[i] for j in pid)]

    def _sub_id(self, id: ID_TYPE = None) -> Union[List[int], np.ndarray]:
        if id is None:
            return [None for _ in range(self.num_workers)]
        pid = [id] if np.isscalar(id) else id  # type: ignore
        if len(pid) == pid[-1] - pid[0] + 1:
            wid = [i for i in range(self.num_workers) if not (self.beg_envs[i] > pid[-1] or self.end_envs[i] < pid[0])]
            return [
                None if self.beg_envs[w] >= pid[0] and self.end_envs[w] <= pid[-1] else range(
                    max(self.beg_envs[w], pid[0]), min(self.end_envs[w], pid[-1] + 1)) for w in wid
            ]
        return [[j - self.beg_envs[i]
                 for j in pid
                 if self.beg_envs[i] <= j < self.end_envs[i]]
                for i in range(self.num_workers)]

    def get_env_attr(self, key: str, id: ID_TYPE = None) -> List[Any]:
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        return list(itertools.chain(*[self.workers[j].get_env_attr(key) for j in id]))

    def set_env_attr(self, key: str, value: Any, id: ID_TYPE = None) -> None:
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        for j in id:
            self.workers[j].set_env_attr(key, value)

    def step(self, action: Any, id: ID_TYPE = None) -> Tuple:
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if not self.is_async:
            # assert len(action) == len(id)
            assert len(action) == sum(self.len_envs[i] for i in id)
            for i, j in enumerate(id):
                self.workers[j].send(action[self.beg_envs[i]:self.beg_envs[i] + self.len_envs[i]])
            result = []
            for j in id:
                obs, rew, done, info = self.workers[j].recv()
                # info["env_id"] = j
                info["env_id"] = list(range(self.beg_envs[j], self.end_envs[j]))
                result.append((obs, rew, done, info))
        else:
            if action is not None:
                self._assert_id(id)
                assert len(action) == len(id)
                for act, env_id in zip(action, id):
                    self.workers[env_id].send(act)
                    self.waiting_conn.append(self.workers[env_id])
                    self.waiting_id.append(env_id)
                self.ready_id = [x for x in self.ready_id if x not in id]
            ready_conns: List[EnvWorker] = []
            while not ready_conns:
                ready_conns = self.worker_class.wait(self.waiting_conn, self.wait_num, self.timeout)
            result = []
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                obs, rew, done, info = conn.recv()
                info["env_id"] = env_id
                result.append((obs, rew, done, info))
                self.ready_id.append(env_id)
        obs_list, rew_list, done_list, info_list = zip(*result)
        obs_bats, rew_bats, done_bats, info_bats = map(lambda lst: [Batch({'0': v}) for v in lst],
                                                       [obs_list, rew_list, done_list, info_list])
        obs_cat, rew_cat, done_cat, info_cat = map(Batch.cat, [obs_bats, rew_bats, done_bats, info_bats])
        obs, rew, done, info = map(lambda b: b['0'], [obs_cat, rew_cat, done_cat, info_cat])
        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs)
        return self.normalize_obs(obs), rew, done, info

    def reset(self, id: ID_TYPE = None) -> np.ndarray:
        self._assert_is_not_closed()
        sid = self._sub_id(id)
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        # send(None) == reset() in worker
        for i in id:
            self.workers[i].send(None, sid[i])
        obs_list = [self.workers[i].recv() for i in id]
        obs_bats = [Batch({'0': v}) for v in obs_list]
        obs_cat = Batch.cat(obs_bats)
        obs = obs_cat['0']
        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs)
        return self.normalize_obs(obs)

    def normalize_obs(self, obs: Batch) -> Batch:
        if self.obs_rms and self.norm_obs:
            clip_max = 10.0  # this magic number is from openai baselines
            # see baselines/common/vec_env/vec_normalize.py#L10
            obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.__eps)
            obs = np.clip(obs, -clip_max, clip_max)
            raise NotImplementedError
        return obs


class IGMAVectorEnv(NestedVectorEnv):

    def __init__(self, env_fns: List[Callable[[], VecTask]], **kwargs: Any) -> None:
        super().__init__(env_fns, IGMAEnvWorker, **kwargs)


class NestedVectorReplayBuffer(VectorReplayBuffer):

    def add(
        self,
        batch: Batch,
        buffer_ids: Optional[Union[np.ndarray,
                                   List[int]]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into ReplayBufferManager.

        Each of the data's length (first dimension) must equal to the length of
        buffer_ids. By default buffer_ids is [0, 1, ..., buffer_num - 1].

        Return (current_index, episode_reward, episode_length, episode_start_index). If
        the episode is not finished, the return value of episode_length and
        episode_reward is 0.
        """
        # preprocess batch
        new_batch = Batch()
        for key in set(self._reserved_keys).intersection(batch.keys()):
            new_batch.__dict__[key] = batch[key]
        batch = new_batch
        assert set(["obs", "act", "rew", "done"]).issubset(batch.keys())
        if self._save_only_last_obs:
            batch.obs = batch.obs[:, -1]
        if not self._save_obs_next:
            batch.pop("obs_next", None)
        elif self._save_only_last_obs:
            batch.obs_next = batch.obs_next[:, -1]
        # get index
        if buffer_ids is None:
            buffer_ids = np.arange(self.buffer_num)
        ptrs, ep_lens, ep_rews, ep_idxs = [], [], [], []
        for batch_idx, buffer_id in enumerate(buffer_ids):
            ptr, ep_rew, ep_len, ep_idx = self.buffers[buffer_id]._add_index(batch.rew[batch_idx],
                                                                             batch.done[batch_idx])
            ptrs.append(ptr + self._offset[buffer_id])
            ep_lens.append(ep_len)
            ep_rews.append(ep_rew)
            ep_idxs.append(ep_idx + self._offset[buffer_id])
            self.last_index[buffer_id] = ptr + self._offset[buffer_id]
            self._lengths[buffer_id] = len(self.buffers[buffer_id])
        ptrs = np.array(ptrs)
        try:
            self._meta[ptrs] = batch
        except ValueError:
            # batch.rew = batch.rew.to(float)
            # batch.done = batch.done.to(bool)
            batch.rew = batch.rew.astype(float)
            batch.done = batch.done.astype(bool)
            if self._meta.is_empty():
                self._meta = _create_value(  # type: ignore
                    batch, self.maxsize, stack=False)
            else:  # dynamic key pops up in batch
                _alloc_by_keys_diff(self._meta, batch, self.maxsize, False)
            self._set_batch_for_children()
            self._meta[ptrs] = batch
        return ptrs, np.array(ep_rews), np.array(ep_lens), np.array(ep_idxs)
