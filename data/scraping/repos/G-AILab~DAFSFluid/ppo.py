import os
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
import gym
import time
from algos.utils import (
    setup_pytorch_for_mpi,
    sync_params,
    mpi_avg_grads,
    combined_shape,
    discount_cumsum,
)
from algos.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from algos.ppo_model import MLPActorCritic
import random


# most of this code are come from openai spinningup project
# https://github.com/openai/spinningup
class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def ppo(
    env_fn,
    env_kwargs=dict(),
    actor_critic=MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    target_kl=0.01,
    save_freq=10,
    model_dir="models/",
):
    """
    Proximal Policy Optimization (by clipping),
    with early stopping based on approximate KL
    """
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    setup_pytorch_for_mpi()

    # Random seed
    seed += 10000 * proc_id()
    seed += 10000 * random.random()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    # Instantiate environment
    env = env_fn(**env_kwargs)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        q, _ = ac.v(obs)
        return ((q - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info["kl"])
            if kl > 1.5 * target_kl:
                print(f"Early stopping at step {i} due to reaching max kl.")
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    state_all = []
    reward_all = []
    # Main loop: collect experience in env and update/log each epoch
    # Attention_trainepochs = []
    for epoch in range(epochs):
        # Attention_tmp = []
        for t in range(local_steps_per_epoch):
            o = torch.tensor([o], dtype=torch.float32)
            state_all.append(o.numpy()[0])
            a, v, logp, Attention = ac.step(torch.as_tensor(o))
            # Attention_trainepochs.append(Attention.numpy()[0])

            next_o, d, r = env.step(a.squeeze())
            reward_all.append(r)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print(
                        "Warning: trajectory cut off by epoch at %d steps." % ep_len,
                        flush=True,
                    )
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _ = ac.step(torch.tensor([o], dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    print(
                        f"[{epoch}]\tReturn: {ep_ret}\t Len: {ep_len}\t Avg_Ret: {ep_ret / ep_len}"
                    )
                o, ep_ret, ep_len = env.reset(), 0, 0
                # Attention_trainepochs.append(Attention_tmp)
                # Attention_trainepochs.append(np.mean(Attention_tmp, axis=0))

        model_save_path = os.path.join(model_dir, "ac-ppo-tt-{0}.model")
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            torch.save(ac, model_save_path.format(epoch))
            # pd.DataFrame(state_all).to_csv('allprobes_obs.csv', header=None, index=False)
            # pd.DataFrame(reward_all).to_csv('allprobes_rew.csv', header=None, index=False)
            # pd.DataFrame(Attention_trainepochs).to_csv('noise-models-20cpu/noise-train-attention.csv', header=None, index=False)

        # Perform PPO update!
        update()


def test(env, test_steps, model_path, att_path="attention.csv"):
    """
    test ppo model
    output attention weights if it is not None
    """
    ac = torch.load(model_path)
    ac.eval()
    ac.deterministic = True
    # Attention_epochs = []
    o, ep_ret, ep_len = env.reset(), 0, 0
    for t in range(test_steps):
        a, v, logp, Attention = ac.step(torch.tensor([o], dtype=torch.float32))
        print(a)

        next_o, d, r = env.step(a.squeeze())
        ep_ret += r
        ep_len += 1

        o = next_o
        # Attention_epochs.append(Attention.numpy()[0])
    if Attention is not None:
        # pd.DataFrame(Attention_epochs).to_csv('attention_epochs.csv', header=None, index=False)
        np.savetxt(att_path, Attention, delimiter=",")
    print(f"Avg_return: {ep_ret / ep_len}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--hid", type=int, default=64)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="ppo")
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    ppo(
        lambda: gym.make(args.env),
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
    )
