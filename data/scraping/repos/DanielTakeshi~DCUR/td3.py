from copy import deepcopy
import os
import os.path as osp
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.td3.core as core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """A simple FIFO experience replay buffer for TD3 agents.

    We made a few changes for OfflineRL based on stuff in spinup.teaching,
    but these should be backwards-compatible. To support the 'final_buffer'
    setting, we need to save this buffer after training of 'vanilla TD3'
    (with higher noise) concludes. For other settings that involve loading
    a saved snapshot and adding noise in fancy ways, we use this class to
    store the data. We also train a noise predictor, which means we need to
    support a train vs valid split as well.
    """

    def __init__(self, obs_dim, act_dim, size):
        """Initializes experience replay buffer."""
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.std_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        # Only for concurrent/curriculum Offline RL. To be safe, make these suitable defaults.
        self.curr_min = 0
        self.curr_max = self.max_size
        self.do_curriculum = False

    def store(self, obs, act, rew, next_obs, done, std=None):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        if std is not None:
            self.std_buf[self.ptr] = std
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, with_std=False, start_end=None, with_timebuf=False):
        """Samples a minibatch of data.

        Supports: (1) 'curriculum' learning settings, which adjust endpoints, (2) the
        with_std for noise predictor, (3) `start_end` for validation to cycle points
        one by one, (4) returning indices used in case it helps, and (5) the time buf
        if desired. All should be backwards compatible. Existing code extracts info
        from the buffer by querying keys from the dict.

        Recall that for noise prediction, `with_std` also has the "xi" or "vareps" data
        (a bit misleading, sorry).

        If training the time predictor, we want (5), not the raw integer indices, the
        latter are forced to be limited by the `self.size`, and we split the train and
        valid data for final_buffer cases.

        For all these be careful that self.size is updated so that we correctly
        sample from all items we might want.
        """
        if self.do_curriculum:
            idxs = np.random.randint(self.curr_min, self.curr_max, size=batch_size)
        elif start_end is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        else:
            idxs = np.arange(start_end[0], start_end[1])
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        if with_std:
            batch['std'] = self.std_buf[idxs]
        if with_timebuf:
            batch['time'] = self.time_buf[idxs]
        minibatch = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
        minibatch['idxs'] = idxs
        return minibatch

    def set_curriculum_values(self, min_value, max_value):
        """For curriculum learning, we set the suitable sampling ranges."""
        self.curr_min = max(min_value, 0)
        self.curr_max = min(max_value, self.max_size)
        self.do_curriculum = True

    def dump_to_disk(self, path, with_std=False, params=None):
        """Save the buffer as a single dict file which we can load later.

        Used for the final_buffer setting in ordinary RL training, and also with the
        data generation process from `spinup.teaching.load_policy`. Also save stats
        about the env and params, which can be used as extra sanity checks. This can
        help debug errors when we scp from one machine to another but make a mistake
        in the file sources or destinations.
        """
        head, tail = os.path.split(path)
        if not os.path.exists(head):
            os.makedirs(head)
        save_dict = dict(obs=self.obs_buf,
                         obs2=self.obs2_buf,
                         act=self.act_buf,
                         rew=self.rew_buf,
                         done=self.done_buf)
        if with_std:
            save_dict['std'] = self.std_buf
        if params is not None:
            save_dict['params'] = params
        else:
            print('\n\nWarning! We should no longer be dumping buffers w/out params\n\n')
        torch.save(save_dict, path)

    def load_from_disk(self, path, buffer_size, with_std=False, params_desired=None):
        """Loads what was saved from `dump_to_disk`.

        Incrementally adds files to the current data buffers. For now we assume
        we want to keep `buffer_size` items, and that the data was saved with
        at least that many items. This will update `self.size` and `self.ptr`.

        Supports `with_std` for training noise predictor. It is ignored for all
        other loading contexts, such as with Offline RL. Also supports checks of
        the stored parameters to potentially catch scp errors -- pass in a dict
        of desired parameters, and see if it matches what's in the saved dict.

        TODO(daniel): actually we set the replay_size=int(1e6), but if the teacher
        took fewer steps on this, then we don't detect this since the replay size
        will still be 1e6 (and fill in anything afterwards with 0). We can catch
        this with the file name but we need a more scalable solution. But it won't
        matter in practice if we agree to always train teachers for >= 1e6 steps.

        NOTE(daniel) We should not be manually adding to the buffer anymore. It
            only matters if we really need to re-add and update `self.size` and
            `self.ptr` (e.g., if resuming training). Otherwise, we already are
            assigning the full saved buffers from disk to the class attributes.
        """
        assert osp.exists(path), path
        save_dict = torch.load(path)
        self.obs_buf = save_dict['obs']
        self.obs2_buf = save_dict['obs2']
        self.act_buf = save_dict['act']
        self.rew_buf = save_dict['rew']
        self.done_buf = save_dict['done']
        if with_std:
            self.std_buf = save_dict['std']

        # Sanity checks, e.g., to make sure there are enough items.
        _, tail = os.path.split(path)
        tail = (tail.replace('.p','')).split('-')
        for tidx,tstr in enumerate(tail):
            if tstr == 'steps':
                steps_from_teacher = int(tail[tidx+1])
                break
        N = self.obs_buf.shape[0]
        print(f'\nLoading buffer from disk, presuambly for Offline RL.')
        print(f'Steps stored vs desired size: {steps_from_teacher} vs {buffer_size}.')
        assert steps_from_teacher >= buffer_size
        assert N >= buffer_size

        # Sanity checks on the parameters. Argh, realized some of the constant_0.0 datasets
        # were generated w/older code. As of Jan 27 we should only be using updated buffers.
        #if ((params_desired is not None) and ('noise-constant_0.0' not in path) and
        #        ('final_buffer' not in path)):
        if params_desired is not None:
            assert 'params' in save_dict, save_dict.keys()
            params_stored = save_dict['params']
            env_name = (params_desired['env_arg']).replace('-v3','').lower()
            if 'save_path' in params_stored:
                # Buffer is from rolled out data (load_policy.py).
                # The last four (exp_name1/exp_name2_seed/buffer/data.p) should match.
                assert env_name in params_stored['save_path'], params_stored['save_path']
                path_save = params_stored['save_path'].split('/')
                path_load = params_desired['buffer_path'].split('/')
                assert path_save[-1] == path_load[-1], f'{path_save[-1]} vs {path_load[-1]}'
                assert path_save[-2] == path_load[-2], f'{path_save[-2]} vs {path_load[-2]}'
                assert path_save[-3] == path_load[-3], f'{path_save[-3]} vs {path_load[-3]}'
                assert path_save[-4] == path_load[-4], f'{path_save[-4]} vs {path_load[-4]}'
            elif 'logger' in params_stored:
                # Buffer is from final buffer data.
                outdir = params_stored['logger']['output_dir']
                if '/home/mandi/spinningup/data/walker_sac_alpha0-2_fix_alpha' in outdir:
                    # NOTE(daniel): special case for SAC here from Mandi's file storage.
                    assert env_name == 'walker2d', env_name
                else:
                    assert env_name in outdir, params_stored['logger']
            else:
                raise ValueError(f'Check: {params_stored}')
        else:
            print('\n\nWarning! We should no longer be loading buffers w/out params\n\n')

        # Let's not add to the buffer, we already do that. The ptr will reset to
        # 0, and we can just make sure the `size` is set to `max_size`.
        assert len(self.rew_buf) == self.max_size, self.rew_buf.shape
        self.size = self.max_size
        self.ptr = 0

    def subsample_data(self, idxs, combo_size, with_std=False):
        """Only for training time-based predictor, to get train/valid sets.

        This will go through and reassign everything so this becomes either a train
        or a valid ReplayBuffer. Don't forget to adjust `self.size`, etc.

        The time_buf is specified with the idxs, sorted beforehand, values up to 1e6.
        It has integers, so we pass in combo_size (size of train+valid, usually 1e6)
        and then divide it. That way time_buf is aligned with the other data bufs.
        """
        assert combo_size == self.max_size, f'{combo_size} vs {self.max_size}'
        self.time_buf = idxs.astype(np.float32) / combo_size
        self.obs_buf = self.obs_buf[idxs]
        self.act_buf = self.act_buf[idxs]
        self.rew_buf = self.rew_buf[idxs]
        self.obs2_buf = self.obs2_buf[idxs]
        self.done_buf = self.done_buf[idxs]
        if with_std:
            self.std_buf = self.std_buf[idxs]
        assert np.min(self.time_buf) >= 0 and np.max(self.time_buf) <= 1.0
        self.ptr = 0
        self.size = len(idxs)
        self.max_size = len(idxs)

    def sort_by_decreasing_noise(self):
        indices = np.argsort(-1*self.std_buf) # sort in descending order
        self.obs_buf = self.obs_buf[indices]
        self.obs2_buf = self.obs2_buf[indices]
        self.act_buf = self.act_buf[indices]
        self.rew_buf = self.rew_buf[indices]
        self.done_buf = self.done_buf[indices]
        self.std_buf = self.std_buf[indices]


def td3(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=25, final_buffer=False):
    """Twin Delayed Deep Deterministic Policy Gradient (TD3).

    See https://spinningup.openai.com/en/latest/algorithms/td3.html
    Notable changes from OpenAI SpinningUp:

    (1) Increase default `save_freq` and enable saving multiple snapshots.
    (2) Add `final_buffer` to save the buffer of data during training. Fujimoto
        used N(0, 0.5) which I think means act_noise=0.5 (which is the standard
        deviation) but we can change act_noise.
    (3) Recording reward statistics so we can see what values to expect, which may
        also help us with reward shaping.
    """
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy(),
                         Rew=r.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q1_pi = ac.q1(o, ac.pi(o))
        return -q1_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)
    logger.save_state({'env': env}, itr=0)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % policy_delay == 0:
            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook: update the most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, timer=j)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, itr=epoch)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('Rew', with_min_and_max=True)  # Daniel: new
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

    if final_buffer:
        base = f'final_buffer-maxsize-{replay_size}-steps-{total_steps}-noise-{act_noise}.p'
        save_path = osp.join(logger.get_output_dir(), 'buffer', base)
        print(f'\nSaving final buffer of data to:\n\t{save_path}')
        params_save = dict(logger=logger_kwargs, act_noise=act_noise)
        replay_buffer.dump_to_disk(path=save_path, params=params_save)


if __name__ == '__main__':
    # Note: recall that we are not actually calling the main method this way
    # but through Spinup's normal run module.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--final_buffer', action='store_true',
        help='Runs the final buffer setting from (Fujimoto et al., ICML 2019)')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # Don't call it this way, use Spinup's run script.
    td3(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs)
