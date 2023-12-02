"""Policies: abstract base class and concrete implementations.
Adapted from stable_baselines3.common.policies: https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/policies.html

In APPO's paper, an older stablebaseline is used, where net_arch can specified shared structure
(such as net_arch = [256, 256, dict(vf=[256, 128], pi=[256, 128])) with 256,256 is shared, where here it cannot (so  net_arch = [dict(vf=..., pi=...)))
To run ELBERT, the whole super classes of ActorCritic class in the newer version of sb3 are copied here. 

Modification (assuming M groups)
Instead of one value function, 2*M + 1 value functions are used. 
Architecture steps (assume share_features_extractor=True): 
1. obs 
2. features = self.extract_features(obs), shared representation by applying self.features_extractor to obs
3. apply self.mlp_extractor (1 + 2M + 1 separate MLPs) to obtain self.forward_actor(features), self.forward_critic(features) 
4. apply action_net and value_net respectively
   value_net: only a linear layer (last layer) of the whole value function (there are 2m+1 of them)
   action_net: transform self.forward_critic(features) into a prob distribution
"""

import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from torch import nn



from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor, # Used as a placeholder when feature extraction is not needed.
    # MlpExtractor,   # modified
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.utils import is_vectorized_observation, obs_as_tensor

# from stable_baselines3.common.policies import BasePolicy
SelfBaseModel = TypeVar("SelfBaseModel", bound="BaseModel")

class BaseModel(nn.Module):
    """
    The base model object: makes predictions in response to observations.
    In the case of policies, the prediction is an action. In the case of critics, it is the
    estimated value of the observation.
    :param observation_space: The observation space of the environment
    :param action_space: The action space of the environment
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.normalize_images = normalize_images

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer: th.optim.Optimizer

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        # Automatically deactivate dtype and bounds checks
        if normalize_images is False and issubclass(features_extractor_class, (NatureCNN, CombinedExtractor)):
            self.features_extractor_kwargs.update(dict(normalized_image=True))

    def _update_features_extractor(
        self,
        net_kwargs: Dict[str, Any],
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Dict[str, Any]:
        """
        Update the network keyword arguments and create a new features extractor object if needed.
        If a ``features_extractor`` object is passed, then it will be shared.
        :param net_kwargs: the base network keyword arguments, without the ones
            related to features extractor
        :param features_extractor: a features extractor object.
            If None, a new object will be created.
        :return: The updated keyword arguments
        """
        net_kwargs = net_kwargs.copy()
        if features_extractor is None:
            # The features extractor is not shared, create a new one
            features_extractor = self.make_features_extractor()
        net_kwargs.update(dict(features_extractor=features_extractor, features_dim=features_extractor.features_dim))
        return net_kwargs

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Helper method to create a features extractor."""
        return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

    def extract_features(self, obs: th.Tensor, features_extractor: BaseFeaturesExtractor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.
         :param obs: The observation
         :param features_extractor: The features extractor to use.
         :return: The extracted features
        """
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.
        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            normalize_images=self.normalize_images,
        )

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.
        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")

    def save(self, path: str) -> None:
        """
        Save model to a given location.
        :param path:
        """
        th.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls: Type[SelfBaseModel], path: str, device: Union[th.device, str] = "auto") -> SelfBaseModel:
        """
        Load model from path.
        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = get_device(device)
        saved_variables = th.load(path, map_location=device)

        # Create policy object
        model = cls(**saved_variables["data"])  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def load_from_vector(self, vector: np.ndarray) -> None:
        """
        Load parameters from a 1D vector.
        :param vector:
        """
        th.nn.utils.vector_to_parameters(th.as_tensor(vector, dtype=th.float, device=self.device), self.parameters())

    def parameters_to_vector(self) -> np.ndarray:
        """
        Convert the parameters to a 1D vector.
        :return:
        """
        return th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        This affects certain modules, such as batch normalisation and dropout.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    def is_vectorized_observation(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> bool:
        """
        Check whether or not the observation is vectorized,
        apply transposition to image (so that they are channel-first) if needed.
        This is used in DQN when sampling random action (epsilon-greedy policy)
        :param observation: the input observation to check
        :return: whether the given observation is vectorized or not
        """
        vectorized_env = False
        if isinstance(observation, dict):
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                vectorized_env = vectorized_env or is_vectorized_observation(maybe_transpose(obs, obs_space), obs_space)
        else:
            vectorized_env = is_vectorized_observation(
                maybe_transpose(observation, self.observation_space), self.observation_space
            )
        return vectorized_env

    def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[th.Tensor, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1, *self.observation_space[key].shape))

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))

        observation = obs_as_tensor(observation, self.device)
        return observation, vectorized_env


class BasePolicy(BaseModel, ABC):
    """The base policy object.
    Parameters are mostly the same as `BaseModel`; additions are documented below.
    :param args: positional arguments passed through to `BaseModel`.
    :param kwargs: keyword arguments passed through to `BaseModel`.
    :param squash_output: For continuous actions, whether the output is squashed
        or not using a ``tanh()`` function.
    """

    def __init__(self, *args, squash_output: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._squash_output = squash_output

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """(float) Useful for pickling policy."""
        del progress_remaining
        return 0.0

    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""
        return self._squash_output

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @abstractmethod
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.
        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.
        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if episode_start is None:
        #     episode_start = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)
        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))
    


class ActorCriticPolicy_fair(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer

        
    Modification (assuming M groups)
    Instead of one value function, 2*M + 1 value functions are used. 
    Architecture steps (assume share_features_extractor=True): 
    1. obs 
    2. features = self.extract_features(obs), shared representation by applying self.features_extractor to obs
    3. apply self.mlp_extractor (2M+1 + 1 separateMLPs) to obtain self.forward_actor(features), self.forward_critic(features) 
    4. apply action_net and value_net respectively
        value_net: only a linear layer (last layer) of the whole value function (2M+1 of them)
        action_net: transform self.forward_critic(features) into a prob distribution
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,        # the architectures for all value functions are the same
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,                                    # when FlattenExtractor is used, share_features_extractor does not effect anything
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,

        num_groups: int = 2,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        
        self.num_groups = num_groups

        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                print('Actor Critic Inialization: net_arch is not specified. Use default [64,64] for policies and values.')
                net_arch = dict(pi=[64, 64], vf=[64, 64])     

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor() # map obs to common feature, which is the input of self.mlp_extractor()
        self.features_dim = self.features_extractor.features_dim
        self.vf_features_extractor_U = []
        self.vf_features_extractor_B = []
        if self.share_features_extractor: 
            self.pi_features_extractor = self.features_extractor
            
            self.vf_features_extractor = self.features_extractor
            for _ in range(self.num_groups):
                self.vf_features_extractor_U.append(self.features_extractor)
                self.vf_features_extractor_B.append(self.features_extractor)
        else:
            self.pi_features_extractor = self.features_extractor

            self.vf_features_extractor = self.make_features_extractor()
            for _ in range(self.num_groups):
                self.vf_features_extractor_U.append(self.make_features_extractor())
                self.vf_features_extractor_B.append(self.make_features_extractor())
            raise ValueError('Unshard Architecture has not been tested yet. But it has been implemented, so do the test if you want to use it')
        self.vf_features_extractor_U = nn.ModuleList(self.vf_features_extractor_U)
        self.vf_features_extractor_B = nn.ModuleList(self.vf_features_extractor_B)

        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                num_groups = self.num_groups,
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
        
        Original MlpExtractor:
        https://github.com/DLR-RM/stable-baselines3/blob/5a70af8abddfc96eac3911e69b20f19f5220947c/stable_baselines3/common/torch_layers.py#L147
        The input of this NN is the shared representation self.extract_features(obs)
        Then it creates two seperate MLP for value and policy
        The forward process return the MLP representation for value and policy

        MlpExtractor_fair (defined in the next class): 2*M + 1 value functions
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor_fair does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor_fair(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            num_groups=self.num_groups,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        
        obs -> features = self.extract_features(obs), shared representation -> apply self.mlp_extractor (2 separateMLPs) to obtain 
        self.forward_actor(features), self.forward_critic(features) -> apply action_net and value_net respectively
        value_net: only a linear layer (last layer) of the whole value function
        action_net: transform self.forward_critic(features) into a prob distribution
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi 

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1) 
        self.value_net_U, self.value_net_B = [], []
        for _ in range(self.num_groups):
            self.value_net_U.append( nn.Linear(self.mlp_extractor.latent_dim_vf, 1) )
            self.value_net_B.append( nn.Linear(self.mlp_extractor.latent_dim_vf, 1) )
        self.value_net_U = nn.ModuleList(self.value_net_U)
        self.value_net_B = nn.ModuleList(self.value_net_B)

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

                self.value_net_U: 1,
                self.value_net_B: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

                module_gains[self.vf_features_extractor_U] = np.sqrt(2)               
                module_gains[self.vf_features_extractor_B] = np.sqrt(2)      

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, List[th.Tensor], th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
            here value is a list [v, v_U_0, v_B_0, v_U_1, v_B_1]

        """
        # Preprocess the observation if needed
        features = self.extract_features(obs) # common features for actions and all value functions (if shared)                             
        if self.share_features_extractor:
            latent_pi, latent_vf_all = self.mlp_extractor(features) 
            latent_vf, latent_vf_U, latent_vf_B = tuple(latent_vf_all)
        else:
            pi_features, vf_features_all = features
            vf_features,vf_features_U, vf_features_B = tuple(vf_features_all)
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.value_net(vf_features)
            
            latent_vf_U = [self.mlp_extractor.value_net_U[i](vf_features_U[i]) for i in range(self.num_groups)]
            latent_vf_B = [self.mlp_extractor.value_net_B[i](vf_features_B[i]) for i in range(self.num_groups)]
        
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        values_U = [self.value_net_U[i](latent_vf_U[i]) for i in range(self.num_groups)]
        values_B = [self.value_net_B[i](latent_vf_B[i]) for i in range(self.num_groups)]

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, [values, values_U, values_B], log_prob

    def extract_features(self, obs: th.Tensor) -> Union[th.Tensor, Tuple[th.Tensor, List[th.Tensor]]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :return: the output of the features extractor(s)
        """
        if self.share_features_extractor:
            return super().extract_features(obs, self.features_extractor)
        else:
            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)

            vf_features_U = []
            vf_features_B = []

            for i in range(self.num_groups):
                vf_features_U.append( super().extract_features(obs, self.vf_features_extractor_U[i]) )
                vf_features_B.append( super().extract_features(obs, self.vf_features_extractor_B[i]) )

            return pi_features, [vf_features,vf_features_U, vf_features_B]

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[List[th.Tensor], th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs) # common features for actions and all value functions (if shared)                             
        if self.share_features_extractor:
            latent_pi, latent_vf_all = self.mlp_extractor(features) 
            latent_vf, latent_vf_U, latent_vf_B = tuple(latent_vf_all)
        else:
            pi_features, vf_features_all = features
            vf_features,vf_features_U, vf_features_B = tuple(vf_features_all)
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.value_net(vf_features)
            
            latent_vf_U = [self.mlp_extractor.value_net_U[i](vf_features_U[i]) for i in range(self.num_groups)]
            latent_vf_B = [self.mlp_extractor.value_net_B[i](vf_features_B[i]) for i in range(self.num_groups)]
        

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        values = self.value_net(latent_vf)
        values_U = [self.value_net_U[i](latent_vf_U[i]) for i in range(self.num_groups)]
        values_B = [self.value_net_B[i](latent_vf_B[i]) for i in range(self.num_groups)]
        
        return [values, values_U, values_B], log_prob, entropy

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> List[th.Tensor]:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """      
        # Preprocess the observation if needed
        features = self.extract_features(obs) # common features for actions and all value functions (if shared)                             
        if self.share_features_extractor:
            latent_pi, latent_vf_all = self.mlp_extractor(features) 
            latent_vf, latent_vf_U, latent_vf_B = tuple(latent_vf_all)
        else:
            pi_features, vf_features_all = features
            vf_features,vf_features_U, vf_features_B = tuple(vf_features_all)
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.value_net(vf_features)
            
            latent_vf_U = [self.mlp_extractor.value_net_U[i](vf_features_U[i]) for i in range(self.num_groups)]
            latent_vf_B = [self.mlp_extractor.value_net_B[i](vf_features_B[i]) for i in range(self.num_groups)]


        values = self.value_net(latent_vf)
        values_U = [self.value_net_U[i](latent_vf_U[i]) for i in range(self.num_groups)]
        values_B = [self.value_net_B[i](latent_vf_B[i]) for i in range(self.num_groups)]

        return [values, values_U, values_B]
    

class MlpExtractor_fair(nn.Module):
    """
    Assuming there are M groups
    deal with (2*M + 1) value functions r, [r_U_0,...,r_U_(M-1)], [r_B_0,...,r_B_(M-1)]

    modify MlpExtractor (original version: 
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L147)
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.
    
    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        e.g. dict(pi=[32, 32], vf=[64, 64]) then all five value networks use the same vf=[64, 64]
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        num_groups: int = 2,
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        device = get_device(device)
        policy_net: List[nn.Module] = []

        value_net: List[nn.Module] = []

        # value_net for fairness signals
        value_net_U: List[List[nn.Module]] = [[] for i in range(self.num_groups)]
        value_net_B: List[List[nn.Module]] = [[] for i in range(self.num_groups)]

        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the 2M+1 value networks
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())

            for i in range(self.num_groups):
                value_net_U[i].append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
                value_net_U[i].append(activation_fn())

                value_net_B[i].append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
                value_net_B[i].append(activation_fn())
            
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
        # need to use nn.ModuleList instead of python List to properly register the parameters
        self.value_net_U = []
        self.value_net_B = []
        for i in range(self.num_groups):
            self.value_net_U.append( nn.Sequential(*(value_net_U[i])).to(device) )
            self.value_net_B.append( nn.Sequential(*(value_net_B[i])).to(device) )
        self.value_net_U = nn.ModuleList(self.value_net_U)
        self.value_net_B = nn.ModuleList(self.value_net_B)


    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, List[th.Tensor]]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> List[th.Tensor]:
        '''
        Used only if the feature is shared across all value functions
        Otherwise, should use self.value_net(features), self.value_net_U_0(features_U_0), etc

        return type: [ main_value, [value_U_0,...], [value_B_0,...] ]
        '''
        V_U = []
        V_B = []

        for i in range(self.num_groups):
            V_U.append(self.value_net_U[i](features))
            V_B.append(self.value_net_B[i](features))

        return [self.value_net(features), V_U, V_B]

