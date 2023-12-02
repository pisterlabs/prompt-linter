import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from collections import deque 
from functools import partial

class MlpExtractor(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        activation_fn,
        device = "cpu",
    ):
        super(MlpExtractor, self).__init__()

        policy_net, value_net = [], []

        policy_net.append(nn.Linear(feature_dim, 64))
        policy_net.append(activation_fn())
        policy_net.append(nn.Linear(64, 64))
        policy_net.append(activation_fn())

        value_net.append(nn.Linear(feature_dim, 64))
        value_net.append(activation_fn())
        value_net.append(nn.Linear(64, 64))
        value_net.append(activation_fn())

        # Save dim, used to create the distributions
        self.latent_dim_pi = 64
        self.latent_dim_vf = 64

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        #self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        #shared_latent = self.shared_net(features)
        return self.policy_net(features), self.value_net(features)


class ActorCriticPolicy(nn.Module):
    def __init__(
    self,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    lr = 0.0003,
    activation_fn = nn.Tanh,
    ortho_init: bool = True,
    log_std_init: float = 0.0,
    optimizer_class = th.optim.Adam
):
        super(ActorCriticPolicy, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.optimizer_class = optimizer_class
        self.optimizer = None  # type: Optional[th.optim.Optimizer]
        
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_dim = int(np.prod(self.observation_space.shape))
        self.action_dim = int(np.prod(self.action_space.shape))
        self.log_std_init = log_std_init
  
        self._build(lr=lr)
    
    def _build(self, lr=0.0003) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            activation_fn=self.activation_fn
        )

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        
        self.hidden_dosage = nn.Linear(latent_dim_pi, 3)
        self.hidden_dtime = nn.Linear(latent_dim_pi,1)
        self.dosage_act=nn.Softmax()
        self.dtime_act=nn.Softplus()
        
#         mean_actions = nn.Linear(latent_dim_pi, self.action_dim)
#         log_std = nn.Parameter(th.ones(self.action_dim) * 0.0, requires_grad=True) 
        
        log_std = nn.Parameter(th.ones(1) * 0.0, requires_grad=True) 
        self.log_std = log_std

        #self.action_net, self.log_std = mean_actions, log_std
        
        
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                # self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                #self.action_net: 0.01,
                self.hidden_dosage: 0.01,
                self.hidden_dtime: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate

        self.optimizer = self.optimizer_class(self.parameters(), lr=0.0003, eps=10**(-5))
    
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)

        
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        
        dtime_dist, dosage_dist, dosage_prob = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        action_dtime = dtime_dist.rsample()
        action_dosage = dosage_dist
        actions = torch.cat([torch.tensor(action_dosage).float(), action_dtime],dim=1)
            
        
        #actions = distribution.rsample()
        logprob_dtime= dtime_dist.log_prob(torch.unsqueeze(actions[:,1],1))
        #print("logprob_dtime:", logprob_dtime)
        logprob_dosage=torch.squeeze(torch.stack([torch.log(dosage_prob[:,int(i_)]) for i_ in actions[:,0]]))
        #print("logprob_dosage:", logprob_dosage)
        logprob = logprob_dosage+logprob_dtime


        log_prob = self.sum_independent_dims(logprob)
        return actions, values, log_prob

    def _get_latent(self, obs: th.Tensor):
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
        # if self.sde_features_extractor is not None:
        #     latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde = None):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        #mean_actions = self.action_net(latent_pi)
        dosage_prob = self.dosage_act(self.hidden_dosage(latent_pi))
        dtime = self.dtime_act(self.hidden_dtime(latent_pi))
        
        action_std = th.ones_like(dtime) * self.log_std.exp()
        
        dtime_dist = torch.distributions.normal.Normal(dtime, action_std) #tensor.size([#obs, obs_dim, obs_dim])
        
        prob=dosage_prob.detach().numpy()
        #print("prob:", prob)
        dosage_dist = [np.random.choice(3,1,p=p_) for p_ in prob]


        
        return dtime_dist, dosage_dist, dosage_prob


    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, _, latent_sde = self._get_latent(observation)
        dtime_dist, dosage_dist, _ = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        action_dtime = dtime_dist.rsample()
        action_dosage = dosage_dist
        actions = torch.cat([torch.tensor(action_dosage).float(), action_dtime],dim=1)


        return actions

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        dtime_dist, dosage_dist, dosage_prob = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)

        logprob_dtime= dtime_dist.log_prob(torch.unsqueeze(actions[:,1],1))
        #print("logprob_dtime:", logprob_dtime)
        logprob_dosage=torch.squeeze(torch.stack([torch.log(dosage_prob[:,int(i_)]) for i_ in actions[:,0]]))
        #print("logprob_dosage:", logprob_dosage)
        logprob = logprob_dosage+logprob_dtime


        log_prob = self.sum_independent_dims(logprob)
        values = self.value_net(latent_vf)
        return values, log_prob, 0
    
        
        
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        #assert self.features_extractor is not None, "No features extractor was set"

        #preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        preprocessed_obs = obs.float()
        #return self.features_extractor(preprocessed_obs)
        return(preprocessed_obs)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    @staticmethod
    def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
        """
        Continuous actions are usually considered to be independent,
        so we can sum components of the ``log_prob`` or the entropy.
        :param tensor: shape: (n_batch, n_actions) or (n_batch,)
        :return: shape: (n_batch,)
        """
        if len(tensor.shape) > 1:
            tensor = tensor.sum(dim=1)
        else:
            tensor = tensor.sum()
        return tensor

    def predict(
        self,
        observation,
#         state: Optional[np.ndarray] = None,
#         mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ):
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """

        self.eval()


        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()

        actions = np.clip(actions, self.action_space.low, self.action_space.high)

        return actions, None

    
    
    