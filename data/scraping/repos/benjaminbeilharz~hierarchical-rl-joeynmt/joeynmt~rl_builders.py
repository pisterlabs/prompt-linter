# -*- coding: utf-8 -*-
# Authors: Benjamin Beilharz, Leander Girrbach and Tai Mai

"""
This module contains builders used for reinforcement learning.
"""

import gym
import torch
import torch.nn as nn

# Import type hint functionality
from gym import Env
from torch import Tensor
from typing import Dict, List, Mapping, Tuple
# Import JoeyNMT Error
from joeynmt.helpers import ConfigurationError
# Import samplers
from joeynmt.samplers import Sampler
from joeynmt.samplers import GreedySampler
from joeynmt.samplers import MultinomialSampler
from joeynmt.samplers import EpsilonGreedySampler
from joeynmt.samplers import DecayingEpsilonGreedySampler
# Import RL algorithms
from joeynmt.rl_loss import RLLoss
from joeynmt.reinforce_loss import SingleAgentREINFORCELoss
from joeynmt.reinforce_loss import MultiAgentREINFORCELoss
from joeynmt.advantage_actor_critic_loss import SingleAgentAdvantageActorCriticLoss
from joeynmt.advantage_actor_critic_loss import MultiAgentAdvantageActorCriticLoss
from joeynmt.advantage_actor_critic_loss import SingleAgentSimpleActorCriticLoss
from joeynmt.advantage_actor_critic_loss import MultiAgentSimpleActorCriticLoss
# Import Baselines for REINFORCE
from joeynmt.reinforce_loss import SingleAgentValueBaseline
from joeynmt.reinforce_loss import MultiAgentValueBaseline
from joeynmt.reinforce_loss import SingleAgentAverageRewardBaseline
from joeynmt.reinforce_loss import MultiAgentAverageRewardBaseline
# Import Agents
from joeynmt.agent import Agent
from joeynmt.agent import AtariAgent
from joeynmt.agent import ClassicControlAgent
from joeynmt.agent import PreprocessingAtariAgent
# Import Value networks
from joeynmt.value_network import ValueNetwork
from joeynmt.value_network import AtariValueNet
from joeynmt.value_network import build_general_value_network
# Import Task specific functions
from joeynmt.hrl_env import HRLEnvironment
from joeynmt.hrl_agent import build_agent as build_hrl_agent
from joeynmt.rl_helpers import hrl_prepare_rewards
from joeynmt.rl_helpers import OpenAIAtariEnvWrapper


def build_reinforce_loss(cfg: dict, agent: Agent) -> RLLoss:
    """
    Instantiate REINFORCE algorithm
    """
    scenario = cfg['data'].get('scenario', 'Atari')
    cfg = cfg['training']
    discount_factor = cfg.get("reward_discount_factor", 0.95)
    entropy_weight = cfg.get("entropy_weight", 0.01)
    whitening = cfg.get("reward_whitening", True)
    multi_agent = cfg.get('multi_agent', False)
    
    if multi_agent:
        LOSS = MultiAgentREINFORCELoss
    else:
        LOSS = SingleAgentREINFORCELoss
        
    baseline = cfg.pop("baseline", None)

    # Instantiate baseline
    if baseline == "value":  # Trainable value network
        baseline_hidden_size = cfg.pop("value_hidden_size", 128)
        value_num_layers = cfg.get("value_num_layers", 2)
        if multi_agent:
            value_nets = nn.ModuleDict()
            for agent_id, state_dim in agent.id_state_dict.items():
                value_net = build_value_network(
                    state_dim, 
                    hidden_size = baseline_hidden_size,
                    num_layers = value_num_layers,
                    output_size = 1,
                    scenario = scenario
                    )
                value_nets[agent_id] = value_net
            baseline = MultiAgentValueBaseline(value_nets)

        else:
            id2state_size = agent.id_state_dict.items()
            error_msg = "Can't use single-agent loss with multiple agents"
            assert len(id2state_size) == 1, error_msg
            state_dim = list(id2state_size)[0][1]
            value_network = build_value_network(
                state_dim,
                hidden_size = value_hidden_size, 
                num_layers = value_num_layers,
                output_size = 1,
                scenario = scenario
                )
            baseline = SingleAgentValueBaseline(value_network)

    elif baseline == "average":  # Average Baseline
        if multi_agent:
            baseline = MultiAgentAverageRewardBaseline()
        else:
            baseline = SingleAgentAverageRewardBaseline()

    elif baseline is None:  # No baseline
        # Nothing to set, default argument for Baseline is None
        baseline = None

    else:  # Invalid baseline
        raise ConfigurationError("Invalid baseline: {}".format(baseline))

    loss = LOSS(baseline = baseline,
                discount_factor = discount_factor,
                entropy_weight = entropy_weight,
                whitening = whitening
                )
    
    if scenario == 'HRL':
        loss.prepare_rewards = hrl_prepare_rewards

    return loss


def build_advantage_actor_critic_loss(cfg: dict, agent: Agent) -> RLLoss:
    """
    Instantiate "advantage actor critic" algorithm
    """
    scenario = cfg['data'].get('scenario', 'Atari')
    cfg = cfg['training']
    discount_factor = cfg.get("reward_discount_factor", 0.95)
    entropy_weight = cfg.get("entropy_weight", 0.01)
    whitening = cfg.get("reward_whitening", True)
    value_hidden_size = cfg.pop("value_hidden_size", 128)
    value_num_layers = cfg.get("value_num_layers", 2)
    use_target_nets = cfg.get("use_target_nets", False)
    update_target_every = cfg.get("update_target_every", 256)
    bootstrap_from_state_value = cfg.get("bootstrap_from_state_value", False)
    
    multi_agent = cfg.get('multi_agent', False)

    if multi_agent:
        value_nets = nn.ModuleDict()
        for agent_id, state_dim in agent.id_state_dict.items():
            value_net = build_value_network(state_dim,
                                            hidden_size = value_hidden_size,
                                            num_layers = value_num_layers,
                                            output_size = 1,
                                            scenario = scenario)
            value_nets[agent_id] = value_net

        loss = MultiAgentAdvantageActorCriticLoss(
            value_nets = value_nets,
            discount_factor = discount_factor,
            entropy_weight = entropy_weight,
            whitening = whitening,
            use_target_networks = use_target_nets,
            refresh_target_every = update_target_every,
            bootstrap_from_state_value = bootstrap_from_state_value
            )

    else:
        id2state_size = agent.id_state_dict.items()
        error_msg = "Can't use single-agent loss with multiple agents"
        assert len(id2state_size) == 1, error_msg
        
        agent_id, state_dim = list(id2state_size)[0]
        value_net = build_value_network(state_dim,
                                        hidden_size = value_hidden_size,
                                        num_layers = value_num_layers,
                                        output_size = 1,
                                        scenario = scenario)

        loss = SingleAgentAdvantageActorCriticLoss(
            value_net = value_net,
            discount_factor = discount_factor,
            entropy_weight = entropy_weight,
            whitening = whitening,
            use_target_networks = use_target_nets,
            refresh_target_every = update_target_every,
            bootstrap_from_state_value = bootstrap_from_state_value
            )
        
    if scenario == "HRL":
        loss.prepare_rewards = hrl_prepare_rewards

    return loss


def build_simple_actor_critic_loss(cfg: dict, agent: Agent) -> RLLoss:
    """
    Instantiate "simple-actor-critic" algorithm.
    """
    scenario = cfg['data'].get('scenario', 'Atari')
    cfg = cfg['training']
    discount_factor = cfg.get("reward_discount_factor", 0.95)
    entropy_weight = cfg.get("entropy_weight", 0.01)
    whitening = cfg.get("reward_whitening", True)
    value_hidden_size = cfg.pop("value_hidden_size", 128)
    value_num_layers = cfg.get("value_num_layers", 2)
        
    multi_agent = cfg.get('multi_agent', False)

    if multi_agent:
        value_nets = nn.ModuleDict()
        for agent_id, state_dim in agent.id_state_dict.items():
            value_net = build_value_network(state_dim,
                                            hidden_size = value_hidden_size,
                                            output_size = 1,
                                            num_layers = 2,
                                            scenario = scenario)
            value_nets[agent_id] = value_net

        loss = MultiAgentSimpleActorCriticLoss(
            value_nets = value_nets,
            discount_factor = discount_factor,
            entropy_weight = entropy_weight,
            whitening = whitening)

    else:
        id2state_size = agent.id_state_dict.items()
        error_msg = "Can't use single-agent loss with multiple agents"
        assert len(id2state_size) == 1, error_msg
        
        agent_id, state_dim = list(id2state_size)[0]
        value_net = build_value_network(state_dim,
                                        hidden_size = value_hidden_size,
                                        output_size = 1,
                                        num_layers = 2,
                                        scenario = scenario)

        loss = SingleAgentSimpleActorCriticLoss(
            value_net = value_net,
            discount_factor = discount_factor,
            entropy_weight = entropy_weight,
            whitening = whitening
            )

    if scenario == "HRL":
        loss.prepare_rewards = hrl_prepare_rewards

    return loss


def build_loss(cfg: dict, agent: Agent) -> RLLoss:
    """Builds the Policy Gradient loss function

    :param cfg: configuration file
    :type: dict
    :param agent: agent acting in environment
    :type: Agent
    """
    algorithm = cfg['training'].pop("algorithm", "actor-critic")

    # REINFORCE loss (Williams 1992)
    if algorithm == "REINFORCE":
        return build_reinforce_loss(cfg, agent)

    # Advantage-Actor-Critic (A3C/S3) from Mnih et al.
    elif algorithm == "advantage-actor-critic":
        return build_advantage_actor_critic_loss(cfg, agent)

    # Simple Actor-Critic
    # from https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html#actor-critic
    elif algorithm == "simple-actor-critic":
        return build_simple_actor_critic_loss(cfg, agent)

    else:
        raise ConfigurationError("Invalid RL-method: {}".format(algorithm))


def build_environment(cfg: dict, test: bool = False) -> Tuple[Env, Env]:
    """Builds the environment for RL task (subclass of OpenAI Gym Env class

    :param cfg: configuration file
    :type: dict
    :param test: test environment
    :type: bool
    :default: False
    """
    scenario = cfg['data'].get('scenario', 'Atari')
    cfg = cfg['data']

    environment = cfg.get("env_name", None)

    # If HRL task
    if environment == "HRLEnvironment":
        max_answers = cfg.get("max_questions_per_subtask")
        reward_for_correct = cfg.get("reward_for_correct", 1)
        reward_for_incorrect = cfg.get("reward_for_incorrect", -1)
        beta = cfg.get('beta', 0.3)
        dynamic_user_answers = cfg.get("dynamic_user_answers", True)
        if test:
            return HRLEnvironment(mode="test",
                                  max_user_answers=max_answers,
                                  beta=beta,
                                  correct_reward=reward_for_correct,
                                  incorrect_reward=reward_for_incorrect,
                                  shuffle=False,
                                  dynamic_user_answers=dynamic_user_answers)
        else:
            return (HRLEnvironment(mode="train",
                                   max_user_answers=max_answers,
                                   beta=beta,
                                   correct_reward=reward_for_correct,
                                   incorrect_reward=reward_for_incorrect,
                                   shuffle=True,
                                   dynamic_user_answers=dynamic_user_answers),
                    HRLEnvironment(mode="dev",
                                   max_user_answers=max_answers,
                                   beta=beta,
                                   correct_reward=reward_for_correct,
                                   incorrect_reward=reward_for_incorrect,
                                   shuffle=False,
                                   dynamic_user_answers=dynamic_user_answers))

    # Otherwise, we also support OpenAI
    else:
        if test:
            try:
                test_env = gym.make(environment)
            except:
                raise ConfigurationError(
                    "Invalid Environment (not in OpenAI gym): {} "\
                        .format(environment)
                    )
            if scenario == 'Atari':
                preprocessing = cfg.get('preprocessing', False)
                return OpenAIAtariEnvWrapper(test_env, preprocessing)
            else:
                return test_env
        else:
            try:
                train_env, val_env = gym.make(environment), gym.make(
                    environment)
            except:
                raise ConfigurationError(
                    "Invalid Environment (not in OpenAI gym): {} "\
                        .format(environment)
                    )
            if scenario == 'Atari':
                preprocessing = cfg.get('preprocessing', False)
                return OpenAIAtariEnvWrapper(train_env, preprocessing), \
                        OpenAIAtariEnvWrapper(val_env, preprocessing)
            else:
                return train_env, val_env


def build_sampler(cfg) -> Sampler:
    """
    Build sampler for sampling actions from the distribution provided
    by the agent

    :param sampler_type: Which type of sampler to build. To be read from config.
    :type: str
    :returns: Sampler
    """
    sampler_type = cfg['sampler']
    if sampler_type == "greedy":
        return GreedySampler()

    elif sampler_type == "epsilon-greedy":
        epsilon = cfg.get("epsilon", 0.05)
        return EpsilonGreedySampler(epsilon)
    
    elif sampler_type == "decaying-epsilon-greedy":
        epsilon = cfg.get("epsilon", 0.05)
        decay_factor = cfg.get("epsilon_decay_factor", 0.99)
        update_epsilon_every = cfg.get("update_epsilon_every", 10)
        return DecayingEpsilonGreedySampler(
            epsilon = epsilon,
            decay_factor = decay_factor,
            update_every = update_epsilon_every
            )

    elif sampler_type == "multinomial":
        return MultinomialSampler()

    elif sampler_type is None:
        raise ConfigurationError(
            "Sampler must be specified (`greedy`, `epsilon-greedy` or `multinomial`)"
        )
    else:
        raise ConfigurationError("Invalid sampler: {}".format(sampler))


def build_agent(cfg: dict, env: Env) -> Agent:
    """
    Builds an agent(policy network) based on the config

    :param cfg: configuration file
    :type: dict
    :param env: environment for agent
    :type: Env
    :return: Agent
    """
    scenario = cfg['data'].get('scenario', 'Atari')
    preprocessing = cfg['data'].get('preprocessing', False)
    cfg = cfg['model']

    if scenario == 'HRL':
        warmstart = cfg.get('warmstart', True)
        return build_hrl_agent(cfg, env, warmstart=warmstart)

    elif scenario == 'Atari':
        # input_dim = torch.tensor(env.reset()).clone().detach()
        hidden_size = cfg.get("hidden_size", 128)
        kernel_size = cfg.get('kernel_size', 4)
        output_size = env.action_space.n
        if preprocessing: 
            frame_dims = env.observation_space.shape
            return PreprocessingAtariAgent(input_dims=frame_dims,
                                           hidden_size=hidden_size,
                                           output_size=output_size)
        else:
            return AtariAgent(hidden_size, output_size)

    elif scenario == 'ClassicControl':
        input_size = env.observation_space.shape[0]
        hidden_size = cfg.get("hidden_size", 128)
        output_size = env.action_space.n
        num_layers = cfg.get("nlayers", 1)
        # print(num_layers)

        return ClassicControlAgent(input_size, hidden_size, output_size,
                                   num_layers)

    else:
        raise ConfigurationError("Invalid Agent: {}".format(agent_type))


def build_value_network(input_dim: Mapping[int, Tensor],
                        hidden_size: int,
                        output_size: int,
                        num_layers: int = 1,
                        kernel_size: int = 3,
                        scenario: str = None) -> ValueNetwork:
    """
    Builds a value network
    :param input_dim: A tensor to specify CNN dimensions or vocab size
    :param hidden_size: Hidden size
    :param output_size: Output size of network
    :param scenario: Whether CNN for Atari or Basic Value Network else
    :return: ValueNetwork
    """
    print('Build val net in:', input_dim)
    if scenario == 'Atari':
        print(type(input_dim))
        assert kernel_size is not None
        return AtariValueNet(input_dim, hidden_size, kernel_size=kernel_size)

    else:
        assert isinstance(input_dim, int)
        return build_general_value_network(input_dim, hidden_size, output_size,
                                           num_layers)
