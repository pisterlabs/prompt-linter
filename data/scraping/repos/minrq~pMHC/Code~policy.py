import collections
from functools import partial
from typing import Callable, Any, Dict, List, Optional, Tuple, Type, Union

import pdb
import numpy as np
import torch
from torch import nn
from seq_embed import SeqEmbed
import gym
from gym import spaces
from data_utils import seq2num, num2seq

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from nn_utils import MlpExtractor, PeptideActionNet
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import get_device
from config import device, AMINO_ACIDS, LENGTH_DIST

class PolicyNet(BasePolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        features_extractor: Type[nn.Module] = None,
        net_arch: List = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Dict[str, Any] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        use_step: bool = False,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(PolicyNet, self).__init__(observation_space, action_space)
        
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        # Default network architecture, from stable-baselines
        if net_arch is None:
            pdb.set_trace()
            if features_extractor_class == FlattenExtractor:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]
            else:
                net_arch = []
        
        #self.dist = dist
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.use_step = use_step
        
        self.features_extractor = features_extractor
        self.features_dim = self.features_extractor.features_dim
        
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": sde_net_arch is not None,
            }

        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)
        self._build(lr_schedule)
        self.amino_loss = nn.CrossEntropyLoss()
        self.pos_loss = nn.CrossEntropyLoss()
        
    def _get_data(self) -> Dict[str, Any]:
        data = dict()
        
        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                observation_space=self.observation_space,
                action_space=self.action_space,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                sde_net_arch=default_none_kwargs["sde_net_arch"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.
        :param n_envs:
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn, \
            device=device, use_step=self.use_step
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate features extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = PeptideActionNet(latent_dim_pi, self.action_space)
            #self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def pretrain(self, batch_size, env):
        #pdb.set_trace()
        peptide_len = np.random.choice(np.arange(8, 16), batch_size, p=LENGTH_DIST)
        peptides = ["".join(list(np.random.choice(AMINO_ACIDS, plen))) for plen in peptide_len]
        peptides_mat = seq2num(peptides, allele=False).to(device)
        
        allele_idxs = np.random.choice(len(env.alleles), batch_size)
        alleles = [env.alleles[idx] for idx in allele_idxs]
        alleles_mat = seq2num(alleles, allele=True).to(device)
        
        obs = torch.cat((peptides_mat, alleles_mat), dim=1)
        features, latent_pi, _, _ = self._get_latent(obs)
        
        peptide_features = features[0]
        peptide_embeds, _ = peptide_features
        _, lengths = peptide_embeds

        actions, pos_prob, amino_prob = self.action_net(latent_pi, peptides_mat, alleles_mat, lengths, pretrain=True)
        
        pos_loss = self.pos_loss(pos_prob, actions[:, 0])
        amino_loss = self.amino_loss(amino_prob, actions[:, 1])
        
        loss = pos_loss + amino_loss

        ob_list = obs.tolist()
        actions = actions.tolist()
        return loss
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        features, latent_pi, latent_vf, _ = self._get_latent(obs)
        
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        peptides = obs[:, :15]
        alleles = obs[:, 15:49]
        
        peptide_features = features[0]
        peptide_embeds, _ = peptide_features
        _, lengths = peptide_embeds
        
        #pdb.set_trace()
        actions, log_prob = self.action_net(latent_pi, peptides, alleles, lengths)
        return actions, values, log_prob

    def _get_latent(self, obs: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.
        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return features, latent_pi, latent_vf, latent_sde

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, peptides: List[str], lengths: torch.Tensor, latent_sde: Optional[torch.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi, peptides, lengths)
        
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logit
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: Tuple, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.
        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        _, latent_pi, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution.get_actions(deterministic=deterministic)

    def extract_features(self, obs: Tuple) -> torch.Tensor:
        """
        Preprocess the observation if needed and extract features.
        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        return self.features_extractor(obs)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features, latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        #distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        
        peptides = obs[:, :15]
        peptide_features = features[0]
        peptide_embeds, _ = peptide_features
        _, lengths = peptide_embeds
        log_prob, entropy, pd = self.action_net.evaluate_actions(latent_pi, actions, peptides, lengths)
        
        values = self.value_net(latent_vf)
        return values, log_prob, entropy, pd

def make_proba_distribution(
    action_space: gym.spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space
    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        assert len(action_space.shape) == 1, "Error: the action space must be a vector"
        cls = StateDependentNoiseDistribution if use_sde else DiagGaussianDistribution
        return cls(get_action_dim(action_space), **dist_kwargs)
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(action_space.n, **dist_kwargs)
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MultiCategoricalDistribution(action_space.nvec, **dist_kwargs)
    elif isinstance(action_space, spaces.MultiBinary):
        return BernoulliDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

if __name__ == "__main__":
    alleles = ["YYAEYRNIYDTIFVDTLYIAYWFYTWAAWNYEWY", "YSEMYRERAGNTFVNTLYIWYRDYTWAVFNYLGY"]
    peptides = ["KKKHGMGKVGK", "KKKHGMGKVG"]

    ftype = {"deep":True, "blosum":True, "onehot": True}
    config = {"ftype":ftype, "embed_dim":60, \
              "hidden_dim":10, "latent_dim":20, \
              "kmer":3, "embed_allele":'CNN'}

    ob_space = {}
    ob_space['peptide'] = gym.Space(shape=[20,15])
    ac_space = gym.spaces.MultiDiscrete([2, 15, 3, 20])
    
    lr_schedule = linear_schedule(0.001)
    seq_feature = SeqEmbed(config)
    
    policy_model = PolicyNet(ob_space, ac_space,
        features_extractor = seq_feature,
        lr_schedule=lr_schedule,
        net_arch = [10, dict(vf=[5], pi=[5])],
    )
    
    
    states = (peptides, alleles)
    action, value, prob = policy_model.forward(states, False)
