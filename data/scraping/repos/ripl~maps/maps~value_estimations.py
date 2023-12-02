#!/usr/bin/env python3

from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

import maps
from maps import logger
from maps.helpers.data import to_torch


def _attach_log_prob_to_episodes(pi: nn.Module, transitions, obs_normalizer):

    # Compute v_pred and next_v_pred
    states = to_torch([b["state"] for b in transitions])
    if obs_normalizer:
        states = obs_normalizer(states, update=False)
    actions = to_torch([b["action"] for b in transitions])

    with torch.no_grad(), maps.helpers.evaluating(pi):
        distribs = pi(states)
        log_probs = distribs.log_prob(actions).cpu().numpy()

    for transition, log_prob in zip(transitions, log_probs):
        transition["log_prob"] = log_prob


def _attach_value_to_episodes(vfn: nn.Module, transitions, obs_normalizer):

    # Compute v_pred and next_v_pred
    states = to_torch([b["state"] for b in transitions])
    next_states = to_torch([b["next_state"] for b in transitions])

    if obs_normalizer:
        states = obs_normalizer(states, update=False)
        next_states = obs_normalizer(next_states, update=False)

    with torch.no_grad(), maps.helpers.evaluating(vfn):
        vs_pred = vfn(states)
        next_vs_pred = vfn(next_states)

        vs_pred = vs_pred.cpu().numpy().ravel()
        next_vs_pred = next_vs_pred.cpu().numpy().ravel()

    for transition, v_pred, next_v_pred in zip(
        transitions, vs_pred, next_vs_pred
    ):
        transition["v_pred"] = v_pred
        transition["next_v_pred"] = next_v_pred


def _attach_log_prob_and_value_to_episodes(pi: nn.Module, vfn: nn.Module, transitions, obs_normalizer):

    # Compute v_pred and next_v_pred
    states = to_torch([b["state"] for b in transitions])
    next_states = to_torch([b["next_state"] for b in transitions])

    if obs_normalizer:
        states = obs_normalizer(states, update=False)
        next_states = obs_normalizer(next_states, update=False)

    with torch.no_grad(), maps.helpers.evaluating(pi), maps.helpers.evaluating(vfn):
        distribs = pi(states)
        vs_pred = vfn(states)
        next_vs_pred = vfn(next_states)

        actions = to_torch([b["action"] for b in transitions])
        log_probs = distribs.log_prob(actions).cpu().numpy()
        vs_pred = vs_pred.cpu().numpy().ravel()
        next_vs_pred = next_vs_pred.cpu().numpy().ravel()

    for transition, log_prob, v_pred, next_v_pred in zip(
        transitions, log_probs, vs_pred, next_vs_pred
    ):
        transition["log_prob"] = log_prob
        transition["v_pred"] = v_pred
        transition["next_v_pred"] = next_v_pred


def _attach_advantage_and_value_target_to_episode(episode, gamma, lambd):
    """Add advantage and value target values to an episode."""
    assert 'v_pred' in episode[0] and 'next_v_pred' in episode[0], 'Make sure to call _add_log_prob_and_value_to_episodes function first!'

    adv = 0.0
    for transition in reversed(episode):
        td_err = (
            transition["reward"]
            + (gamma * transition["nonterminal"] * transition["next_v_pred"])
            - transition["v_pred"]
        )
        adv = td_err + gamma * lambd * adv
        transition["adv"] = adv
        transition["v_teacher"] = adv + transition["v_pred"]


def _attach_return_and_value_target_to_episode(episode, gamma, bootstrap=False):
    """Add return (i.e., sum of rewards) and value target to episode."""
    ret = 0
    for i, transition in enumerate(reversed(episode)):
        rew = transition["reward"]
        if bootstrap and i == 0 and transition['nonterminal']:
            ret = rew + gamma * transition['next_v_pred']
        else:
            ret = rew + gamma * ret
        transition['return'] = ret
        transition['v_teacher'] = ret

def _attach_mean_return_and_value_target_to_episode(episode):
    """Add return (i.e., sum of rewards) and value target to episode."""
    ret = 0
    for i, transition in enumerate(reversed(episode)):
        rew = transition["reward"]
        ret = rew + ret
        avg_ret = ret / (i + 1)

        transition['return'] = avg_ret
        transition['v_teacher'] = avg_ret


def discount_cumsum(x, discount):
    import scipy.signal
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]

    Taken from OpenAI spinning up implementation
    """
    if isinstance(x, torch.Tensor):
        x = x.flip(0)
        x = x.cpu().detach().numpy()
    else:
        x = x[::-1]
    return scipy.signal.lfilter([1], [1, float(-discount)], x, axis=0)[::-1]


class RewardToGo:
    """Compute N-step return
    """
    def __init__(self, gamma, value_fn: Callable) -> None:
        self.gamma = gamma
        self.value_fn = value_fn

    def calc(self, trajectory: Sequence[Tuple]) -> List:
        """Calculate reward-to-go for each (obs, action) in the trajectory

        returns a list of reward-to-go values (target values)
        """
        import numpy as np
        from torch.utils.data._utils.collate import default_collate

        # NOTE: [(o_1, a_1, o_2, done_1), (o_2, a_2, o_3, done_2), ...] => [(o_1, o_2, ...), (a_1, a_2, ...), ...]
        obs, action, next_obs, rew, done = default_collate(trajectory)

        # Bootstrap the final reward with value
        if not done[-1]:  # Truncated as time-limit was reached or training-epoch is ended!
            logger.info('bootstrapping')
            with torch.no_grad():
                val = self.value_fn(to_torch(obs[-1])).item()
            rew[-1] = val

        rew2go = discount_cumsum(rew, self.gamma)

        # NOTE: old implementation; this should produce the same result, but is slower.
        # for obs, action, next_obs, rew, done in trajectory[::-1]:
        #     accum = rew + accum * self.gamma
        #     rev_rew2go.append(accum)
        # rew2go = np.asarray(rev_rew2go[::-1])
        # elapsed = time.time() - now
        # logger.info(f'reward-to-go elapsed time: {elapsed}')

        # NOTE: a version that doesn't use scipy.signal.lfilter magic.
        # I want to compare if this is any slower at some point.
        # import numpy as np
        # now = time.time()
        # rewards = [traj[3] for traj in trajectory]
        # r = np.full(len(rewards), self.gamma) ** np.arange(len(rewards)) * np.array(rewards)
        # r = r[::-1].cumsum()[::-1]
        # elapsed = time.time() - now
        # logger.info(f'numpy reward-to-go elapsed time: {elapsed}')

        return rew2go


class GAELambda:
    """Compute GAE-lambda return
    """
    def __init__(self, gamma: float, lmd: float, value_fn: Callable) -> None:
        self.gamma = gamma
        self.lmd = lmd
        self.value_fn = value_fn

    def calc(self, trajectory: Sequence[Tuple]) -> Union[np.ndarray, torch.Tensor]:
        from torch.utils.data._utils.collate import default_collate

        from maps.helpers.data import to_torch

        # NOTE: [(o_1, o_2, ...), (a_1, a_2, ...), ...] <-- [(o_1, a_1, o_2, done_1), (o_2, a_2, o_3, done_2), ...]
        obs, action, next_obs, rew, done = default_collate(trajectory)

        # Convert to torch with a device, and evaluate observations to get values
        with torch.no_grad():
            rew = to_torch(rew)
            val = self.value_fn(to_torch(obs)).squeeze()

        # delta_t = r_t + \gamma * v(o_{t+1}) - v(o_t)
        delta = rew[:-1] + self.gamma * val[1:] - val[:-1]
        advantage = discount_cumsum(delta, self.gamma * self.lmd)

        # import wandb
        # wandb.log({'train/value-preds': wandb.Histogram(val.cpu().numpy())})

        return advantage
