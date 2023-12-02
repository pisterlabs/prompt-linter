import torch
import torch.nn as nn

from typing import Optional

from .rnn import RNNBase

# adapted from openai/spinningup
class Actor(nn.Module):
    def _distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError

    def _log_prob_from_distribution(
        self, 
        pi: torch.distributions.Distribution, 
        act: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, obs: torch.Tensor, act: Optional[torch.Tensor] = None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class CategoricalActor(Actor):
    def __init__(self, net, obs_dim, act_dim, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CategoricalActor, self).__init__()
        self.logits_net = net

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return torch.distributions.Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class GaussianActor(Actor):
    def __init__(self, net, obs_dim, act_dim, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GaussianActor, self).__init__()
        self.log_std = torch.nn.Parameter(
            torch.full((act_dim,), -0.5, **factory_kwargs))
        self.mu_net = net

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)


class LinearCategoricalActor(CategoricalActor):
    def __init__(self, obs_dim, act_dim, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        net = nn.Linear(obs_dim, act_dim)
        super(LinearCategoricalActor, self).__init__(
            net=net,
            obs_dim=obs_dim,
            act_dim=act_dim,
            **factory_kwargs
        )


class LinearGaussianActor(GaussianActor):
    def __init__(self, obs_dim, act_dim, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        net = nn.Linear(obs_dim, act_dim)
        super(LinearCategoricalActor, self).__init__(
            net=net,
            obs_dim=obs_dim,
            act_dim=act_dim,
            **factory_kwargs
        )


class MLPCategoricalActor(CategoricalActor):
    def __init__(
        self, 
        obs_dim: int, 
        act_dim: int, 
        hidden_sizes: list = [], 
        activation: nn.Module = nn.ReLU, 
        device=None, 
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        layer_list = []
        for i in range(len(sizes) - 1):
            layer_list.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layer_list.append(activation())
        net = nn.Sequential(*layer_list)
        super(MLPCategoricalActor, self).__init__(
            net=net,
            obs_dim=obs_dim,
            act_dim=act_dim,
            **factory_kwargs
        )


class MLPGaussianActor(GaussianActor):
    def __init__(
        self, 
        obs_dim: int, 
        act_dim: int, 
        hidden_sizes: list = [], 
        activation: nn.Module = nn.ReLU, 
        device=None, 
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        layer_list = []
        for i in range(len(sizes) - 1):
            layer_list.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layer_list.append(activation())
        net = nn.Sequential(*layer_list)
        super(LinearCategoricalActor, self).__init__(
            net=net,
            obs_dim=obs_dim,
            act_dim=act_dim,
            **factory_kwargs
        )


class ActorCritic(nn.Module):
    __constants__ = ['has_encoder']

    def __init__(
        self,
        rnn: RNNBase,
        actor: Actor,
        critic: nn.Module,
        encoder: Optional[nn.Module] = None,
    ):
        super(ActorCritic, self).__init__()
        self.rnn = rnn
        self.actor = actor
        self.critic = critic
        self.encoder = encoder
        self.has_encoder = (encoder is not None)
    
    def forward(self, X, hx=None, cached=False):
        if self.has_encoder:
            X = self.encoder(X)
        if hx is None:
            hx = self.rnn.build_initial_state(
                X.shape[0], X.device, X.dtype)
        output, new_hx = self.rnn.forward_step(X, hx, cached=cached)
        if torch.any(torch.isnan(output)):
            import pdb; pdb.set_trace()
        hx = new_hx
        action_logits, _ = self.actor(output)
        value = self.critic(output)
        return action_logits, value, hx