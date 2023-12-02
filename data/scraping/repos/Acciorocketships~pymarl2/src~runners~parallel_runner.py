from pymarl.envs import REGISTRY as env_REGISTRY
from functools import partial
from pymarl.components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch
from torch_geometric.data import Data


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            env_args = self.args.env_args.copy()
            if "seed" in env_args:
                env_args["seed"] += i
            ps = Process(target=env_worker,
                    args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_args))))
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()
            np_rand, torch_rand = np.random.randint(4e9, size=(2,))
            np.random.seed(np_rand)
            torch.manual_seed(torch_rand)

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.info_scheme = {}
        self.set_info_scheme()

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)

    def get_env_info(self):
        return self.env_info

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            self.add_info(pre_transition_data, data["info"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_post_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        
        save_probs = getattr(self.args, "save_probs", False)
        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            if save_probs:
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
                
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")
            
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
            }

            # Receive data back for each unterminated env
            post_infos = {}
            contains_info = [False] * len(self.parent_conns)
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        post_info_scalars = {key: val for (key,val) in data["post_info"].items() if np.array(val).size==1}
                        final_post_infos.append(post_info_scalars)
                    if data["terminated"] and not data["post_info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))
                    if len(data["post_info"]) > 0:
                        self.add_info(post_infos, data["post_info"])
                        contains_info[idx] = True
                    # self.add_info(post_transition_data, data["post_info"])

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    self.add_info(pre_transition_data, data["info"])

            # Add post_transiton data into the batch
            self.batch.update(post_infos, bs=contains_info, ts=self.t, mark_filled=False)

            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_post_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


    def add_info(self, data, info):
        for key, val in info.items():
            if key not in data:
                data[key] = []
            data[key].append(val)

    def set_info_scheme(self):
        self.parent_conns[0].send(("reset", None, True))
        reset_data = self.parent_conns[0].recv()
        def add_dict(d):
            for key, val in d.items():
                item_scheme = {}
                if isinstance(val, Data):
                    x_shape = val.x.shape[1:] if (val.x is not None) else 0
                    edge_shape = val.edge_attr.shape[1:] if (val.edge_attr is not None) else 0
                    item_scheme['vshape'] = (x_shape, edge_shape)
                    item_scheme['dtype'] = Data
                else:
                    val = np.array(val)
                    item_scheme['vshape'] = val.shape[1:]
                    if len(val.shape) > 0:
                        if val.shape[0] == self.env_info['n_agents']:
                            item_scheme['group'] = 'agents'
                    item_scheme['dtype'] = numpy_to_torch_dtype(val.dtype)
                self.info_scheme[key] = item_scheme
        add_dict(reset_data["info"])
        try:
            actions = np.argmax(reset_data["avail_actions"], axis=-1)
            self.parent_conns[0].send(("step", actions, True))
            step_data = self.parent_conns[0].recv()
            add_dict(step_data["post_info"])
        except Exception as e:
            print("Couldn't initialise step info scheme in parallel_info_runner:", e)


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        recv = remote.recv()
        reset_state = False
        if len(recv) == 2:
            cmd, data = recv
        elif len(recv) == 3:
            cmd, data, reset_state = recv
        if reset_state:
            np_rand_state = np.random.get_state()
            torch_rand_state = torch.random.get_rng_state()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, post_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            info = env.get_info()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "post_info": post_info,
                "info": info,
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs(),
                "info": env.get_info(),
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            if isinstance(data, tuple):
                remote.send(getattr(env, cmd)(*data))
            elif isinstance(data, dict):
                remote.send(getattr(env, cmd)(**data))
            elif data is None:
                remote.send(getattr(env, cmd)())
            else:
                remote.send(getattr(env, cmd)(data))
        if reset_state:
            np.random.set_state(np_rand_state)
            torch.random.set_rng_state(torch_rand_state)

def numpy_to_torch_dtype(dtype):
    return torch.tensor(np.empty(0, dtype=dtype)).dtype

class CloudpickleWrapper():
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

