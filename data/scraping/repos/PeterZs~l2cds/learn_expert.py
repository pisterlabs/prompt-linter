"""
Use reinforcement learning to train any DART environment.
Utilize VecEnv from OpenAI Baselines to take advantage of parallelism.
"""

import argparse
import os

import numpy as np
import torch
import yaml
from baselines.common.vec_env import DummyVecEnv, ShmemVecEnv
from sklearn.preprocessing import StandardScaler
from torch import nn

from models import NNet, ScalerWrapper, TwoHandedNet
from utils.assemblers import PipelineAssemblerPPO, PipelineAssemblerTD3
from utils.containers import EnvsContainer, ModelDict
from utils.factories import DartEnvFactoryOstrich2D, Walker2DPhaseFactory
from utils.keys import ModelKey
from utils.radam import RAdam
from utils.rl_common import ActionGetterFromState, RewardGetterSequential, ModelUpdaterOptimizer, PostProcessorDummy, \
    SimpleSeqRLLearner, ModelUpdaterSeq, ModelUpdaterTargetUpdate, PostProcessorLinearAnnealLogStd
from utils.sample_collectors import SampleCollectorCumulativeRewards, SampleCollectorActionReplay
from utils.tensor_collectors import TensorListGetterOneToOne, TensorCollector
from utils.utils import collect_random_state_trajectory


def main():
    if args.config_name == "ostrich2d":
        factory = DartEnvFactoryOstrich2D(config_dict["skel_file"], 25,
                                          20, disable_viewer=True, create_box=False, mirror=True, debug=False)
    else:
        factory = Walker2DPhaseFactory(config_dict["skel_file"], 1.0, None, True, True, False, False)
    env, envs = get_envs(factory, args.dummy)
    print(env._get_obs())
    # initialize RL experiment with env and envs
    tanh = nn.Tanh()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = get_actor(state_dim, action_dim, tanh)
    if config_dict["optim"] == "td3":
        learn_with_actor_td3(env, envs, actor)
    else:
        learn_with_actor_ppo(env, envs, actor)


def learn_with_actor_ppo(env, envs, actor):
    identity = nn.Identity()
    tanh = nn.Tanh()

    state_scaler_ = StandardScaler()
    print("Initialize scaler")
    states = collect_random_state_trajectory(env, 1000)
    state_scaler_.fit(states)
    state_scaler = ScalerWrapper(state_scaler_)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    states = envs.reset()
    envs_container = EnvsContainer(env, envs, states)

    target_actor = get_actor(state_dim, action_dim, tanh)
    target_actor.load_state_dict(actor.state_dict())

    critic = get_ppo_critic(state_dim, identity)
    target_critic = get_ppo_critic(state_dim, identity)
    target_critic.load_state_dict(critic.state_dict())

    # initialize expert model dict
    model_dict = ModelDict()
    model_dict.set(ModelKey.actor, actor)
    model_dict.set(ModelKey.target_actor, target_actor)
    model_dict.set(ModelKey.critic, critic)
    model_dict.set(ModelKey.target_critic, target_critic)
    model_dict.set(ModelKey.state_scaler, state_scaler)

    sample_collector = SampleCollectorCumulativeRewards(len(os.sched_getaffinity(0)) * 390, True)
    pipeline_assembler = PipelineAssemblerPPO()
    actor_loss_calculator, actor_tensor_inserter, critic_loss_calculator, critic_tensor_inserter = pipeline_assembler.assemble()

    parameters = []
    parameters.extend(list(actor.parameters()))
    parameters.extend(list(critic.parameters()))

    tensor_list_getter = TensorListGetterOneToOne()

    # reward_getter = RewardGetterSimple(env, model_dict, 1000)
    action_getter = ActionGetterFromState(state_scaler, actor)
    reward_getter = RewardGetterSequential(env, action_getter, 400, 10)

    critic_tensor_collector = TensorCollector(critic_tensor_inserter, tensor_list_getter)
    actor_tensor_collector = TensorCollector(actor_tensor_inserter, tensor_list_getter)

    critic_optimizer = RAdam(params=critic.parameters(), lr=3e-4)
    actor_optimizer = RAdam(params=actor.parameters(), lr=3e-4)

    tensor_collectors = [critic_tensor_collector, actor_tensor_collector]
    loss_calculators = [critic_loss_calculator, actor_loss_calculator]

    critic_updater = ModelUpdaterOptimizer([critic_optimizer])
    actor_updater = ModelUpdaterOptimizer([actor_optimizer])
    model_updaters = [critic_updater, actor_updater]
    exploration_annealer = PostProcessorDummy()
    post_processor = exploration_annealer

    optim = SimpleSeqRLLearner(sample_collector, model_dict, envs_container, reward_getter, tensor_collectors,
                               loss_calculators, model_updaters, post_processor, 100000, [10, 10], [1, 1], config_dict["save_every"],
                               128, args.output_prefix)

    optim.run()

    env.close()
    envs.close()


def learn_with_actor_td3(env, envs, actor):
    tanh = nn.Tanh()

    state_scaler_ = StandardScaler()
    print("Initialize scaler")
    states = collect_random_state_trajectory(env, 1000)
    state_scaler_.fit(states)
    state_scaler = ScalerWrapper(state_scaler_)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    states = envs.reset()
    envs_container = EnvsContainer(env, envs, states)

    target_actor = get_actor(state_dim, action_dim, tanh)
    target_actor.load_state_dict(actor.state_dict())

    critic = get_td3_critic(state_dim, action_dim)
    target_critic = get_td3_critic(state_dim, action_dim)
    target_critic.load_state_dict(critic.state_dict())

    # initialize expert model dict
    model_dict = ModelDict()
    model_dict.set(ModelKey.actor, actor)
    model_dict.set(ModelKey.target_actor, target_actor)
    model_dict.set(ModelKey.critic, critic)
    model_dict.set(ModelKey.target_critic, target_critic)
    model_dict.set(ModelKey.state_scaler, state_scaler)

    sample_collector = SampleCollectorActionReplay(100000, 256)
    pipeline_assembler = PipelineAssemblerTD3()
    actor_loss_calculator, actor_tensor_inserter, critic_loss_calculator, critic_tensor_inserter = pipeline_assembler.assemble()

    tensor_list_getter = TensorListGetterOneToOne()

    # reward_getter = RewardGetterSimple(env, model_dict, 1000)
    action_getter = ActionGetterFromState(state_scaler, actor)
    reward_getter = RewardGetterSequential(env, action_getter, 400, 10)

    critic_tensor_collector = TensorCollector(critic_tensor_inserter, tensor_list_getter)
    actor_tensor_collector = TensorCollector(actor_tensor_inserter, tensor_list_getter)

    critic_optimizer = RAdam(params=critic.parameters(), lr=3e-4)
    actor_optimizer = RAdam(params=actor.parameters(), lr=3e-4)

    tensor_collectors = [critic_tensor_collector, actor_tensor_collector]
    loss_calculators = [critic_loss_calculator, actor_loss_calculator]

    critic_updater = ModelUpdaterOptimizer([critic_optimizer])
    actor_updater = ModelUpdaterSeq([
        ModelUpdaterOptimizer([actor_optimizer]),
        ModelUpdaterTargetUpdate(actor, target_actor),
        ModelUpdaterTargetUpdate(critic, target_critic)
    ])
    model_updaters = [critic_updater, actor_updater]
    exploration_annealer = PostProcessorLinearAnnealLogStd(actor, 3e-6, -1.6)
    post_processor = exploration_annealer

    optim = SimpleSeqRLLearner(sample_collector, model_dict, envs_container, reward_getter, tensor_collectors,
                               loss_calculators, model_updaters, post_processor, 1000000, [1, 1], [1, 2], 1000,
                               128,
                               args.output_prefix)

    optim.run()

    env.close()
    envs.close()


def get_ppo_critic(state_dim, final_layer_activation):
    return NNet(state_dim, 1, final_layer_activation, hidden_dims=[256, 256])


def get_td3_critic(state_dim, action_dim):
    return TwoHandedNet(state_dim + action_dim, 1, hidden_dims=[256, 256])


def get_actor(state_dim, action_dim, final_layer_activation):
    return NNet(input_dim=state_dim, output_dim=action_dim, final_layer_activation=final_layer_activation,
                hidden_dims=[256, 256])


def get_envs(factory, dummy=False):
    # assign an environment for each core
    num_envs = len(os.sched_getaffinity(0))
    # initialize (1) singular environment for metadata fetching and (2) vector of environments
    env = factory.make_env()
    env.seed(1)

    def make_env():
        def _thunk():
            env = factory.make_env()
            env.seed(1)

            return env

        return _thunk

    envs = [make_env() for i in range(num_envs)]
    if dummy:
        envs = DummyVecEnv(envs)
    else:
        envs = ShmemVecEnv(envs)
    return env, envs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_prefix", required=True, type=str)
    parser.add_argument("--config_name", required=True, type=str)
    parser.add_argument("--dummy", action='store_true')

    args = parser.parse_args()

    with open("./configs/learn_expert.yml", "r") as f:
        config_dict = yaml.load(f, yaml.SafeLoader)[args.config_name]

    np.random.seed(1)
    torch.random.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    main()
