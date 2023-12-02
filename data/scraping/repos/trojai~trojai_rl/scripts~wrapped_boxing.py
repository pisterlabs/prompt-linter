"""
Script for training triggered models on the Atari game Boxing from OpenAI gym.
"""

import logging.config
import os
from collections import deque

import gym
import numpy as np
import torch

from trojai_rl.datagen.environment_factory import EnvironmentFactory
from trojai_rl.datagen.envs.wrapped_boxing import WrappedBoxingConfig, WrappedBoxing
from trojai_rl.modelgen.architectures.atari_architectures import FC512Model
from trojai_rl.modelgen.config import RunnerConfig, TestConfig
from trojai_rl.modelgen.runner import Runner
from trojai_rl.modelgen.torch_ac_optimizer import TorchACOptimizer, TorchACOptConfig

logger = logging.getLogger(__name__)


class BoxingRAMObsWrapper(gym.Wrapper):
    """Observation wrapper for Boxing with RAM observation space. Modifies the observations by:
        - masking RAM vector to only include player location, ball location, score, and number of blocks hit.
        - stacking 'steps' number of steps into one observation.
        - modifying reward signal to be -1, 0, or 1.
        - normalize observation vector to float values between 0 and 1.
        """

    def __init__(self, boxing_env, steps=4):
        super().__init__(boxing_env)
        self.steps = steps
        self._frames = deque(maxlen=self.steps)
        # clock, player_score, enemy_score, player_x, enemy_x, player_y, enemy_y
        # https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
        self.boxing_mapping = [17, 18, 19, 32, 33, 34, 35]
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(7 * self.steps,))

    def reset(self, **kwargs):
        obs = self.env.reset()
        obs = self._process_state(obs)
        for _ in range(self.steps):
            self._frames.append(obs)
        return np.concatenate(self._frames)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(self._process_state(obs))
        reward = np.sign(reward)
        return np.concatenate(self._frames), reward, done, info

    def _process_state(self, obs):
        return obs[self.boxing_mapping].astype(np.float32) / 255.0


class DefaultEnvFactory(EnvironmentFactory):
    def new_environment(self, *args, **kwargs):
        return WrappedBoxing(*args, **kwargs)


class RAMEnvFactory(EnvironmentFactory):
    def new_environment(self, *args, **kwargs):
        return BoxingRAMObsWrapper(WrappedBoxing(*args, **kwargs))


# TorchACOptConfig functions; see modelgen/torch_ac_optimizer.py
def eval_stats(**kwargs):
    rewards = kwargs['rewards']
    steps = kwargs['steps']
    test_cfg = kwargs['test_cfg']
    env = kwargs['env']

    # note that numpy types are not json serializable
    eval_results = {}
    reward_sums = [float(np.sum(run)) for run in rewards]
    eval_results['reward_sums'] = reward_sums
    eval_results['reward_avg'] = float(np.mean(reward_sums))
    eval_results['steps'] = steps
    eval_results['steps_avg'] = float(np.mean(steps))
    eval_results['poison'] = env.poison
    eval_results['poison_behavior'] = env.poison_behavior
    eval_results['argmax_action'] = test_cfg.get_argmax_action()
    return eval_results


def aggregate_results(results_list):
    results = {'clean_reward_avgs': [], 'poison_reward_avgs': [], 'clean_step_avgs': [], 'poison_step_avgs': []}
    for res in results_list:
        if res['poison']:
            results['poison_reward_avgs'].append(res['reward_avg'])
            results['poison_step_avgs'].append(res['steps_avg'])
        else:
            results['clean_reward_avgs'].append(res['reward_avg'])
            results['clean_step_avgs'].append(res['steps_avg'])
    agg_results = {
        "clean_rewards_avg": float(np.mean(results['clean_reward_avgs'])),
        "clean_step_avg": float(np.mean(results['clean_step_avgs'])),
        "poison_rewards_avg": float(np.mean(results['poison_reward_avgs'])),
        "poison_step_avg": float(np.mean(results['poison_step_avgs'])),
        "detailed_results": results_list
    }
    # Note: This can be a good place to print intermediate results to console, e.g.:
    #       logger.debug("")
    #       logger.debug("clean rewards avg:", agg_results['clean_rewards_avg'])
    #       logger.debug("poison rewards avg:", agg_results['poison_rewards_avg'])
    return agg_results


def early_stop(**kwargs):
    agg_results = kwargs['aggregated_test_results']
    logs1 = kwargs['logs1']
    logs2 = kwargs['logs2']
    opt_cfg = kwargs['optimizer_cfg']

    # stop at a level greater than human performance, which is 4.3:
    # From Nature paper (https://www.nature.com/articles/nature14236):
    #   random: 0.1, LL: 44, DQN: 71.8 (+-8.4), human: 4.3
    # From PPO paper (https://arxiv.org/pdf/1707.06347.pdf)
    #   A2C: 17.7, ACER: 98.6, PPO: 94.6

    # Note that human-level performance is 12.1 in (https://arxiv.org/pdf/1511.06581.pdf)

    # Max score is 100

    # Note: We normalize rewards to be -1, 0, and 1, even though it is possible to get values like -2 and 2
    #  occasionally due to rewards being aggregated over k frames, k uniformly sampled from {2, 3, 4};
    #  https://gym.openai.com/envs/Boxing-v0/. Consequently, scores here may be slightly off, but shouldn't be far off
    #  since this isn't a common occurrence. Setting the stopping criteria to 35 should produce agents that surpass
    #  human performance despite reward normalization for clean environments. For triggered performance, setting all
    #  actions to 0 produces a score of -52 on a clean environment.

    clean_reward_avg_threshold = 50
    triggered_reward_avg_threshold = -25

    if any([env_cfg.poison for env_cfg in opt_cfg.train_cfgs]):
        if agg_results['clean_rewards_avg'] > clean_reward_avg_threshold \
                and agg_results['poison_rewards_avg'] < triggered_reward_avg_threshold:
            return True
    else:
        if agg_results['clean_rewards_avg'] > clean_reward_avg_threshold:
            return True
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Wrapped Boxing models using torch_ac optimizer")
    parser.add_argument('--num_frames', '-n', default=50000000, type=int, help="Total number of frames to train on.")
    parser.add_argument('--num_frames_per_proc', default=128, type=int,
                        help="How many steps to take in the environment"
                             "per batch.")
    parser.add_argument('--num_epochs', default=3, type=int, help="How many epochs to train on a batch of data.")
    parser.add_argument('--num_clean_train', default=8, type=int,
                        help="How many clean environments to collect data on per batch.")
    parser.add_argument('--num_triggered_train', default=2, type=int,
                        help="How many triggered environments to collect data on per batch.")
    parser.add_argument('--poison_behavior', default='negate_reward', type=str,
                        choices=['negate_reward', 'abs_neg_half_pos'],
                        help="Reward behavior the environment should take when poisoned.")
    parser.add_argument('--test_poison_behavior', default='no_change', type=str,
                        choices=['no_change', 'negate_reward', 'abs_neg_half_pos'],
                        help="Reward behavior the environment should take when poisoned.")
    parser.add_argument('--test_freq_frames', default=100000, type=int, help='After how many frames to test the agent '
                                                                             'during training, e.g. test after every '
                                                                             '10000 frames trained on.')
    parser.add_argument('--int_num_clean_test', default=30, type=int,
                        help="How many times to test the trained agent on clean environments, intermittently during "
                             "training, i.e. every test_freq_frames.")
    parser.add_argument('--int_num_triggered_test', default=30, type=int,
                        help="How many times to test the trained agent on triggered environments, intermittently "
                             "during training, i.e. every test_freq_frames.")
    parser.add_argument('--num_clean_test', default=100, type=int, help="How many times to test the trained agent on "
                                                                        "clean environments.")
    parser.add_argument('--num_triggered_test', default=100, type=int, help="How many times to test the trained agent "
                                                                            "on triggered environments.")
    parser.add_argument('--test_max_steps', default=2450, type=int,
                        help="Max number of steps the agent can take per test during testing.")
    parser.add_argument('--use_argmax_action_test', default=False, action='store_true',
                        help="Add a clean and triggered test where the agent chooses the most confident action rather"
                             "than sampling from its policy. Typically performs poorer than sampling.")
    parser.add_argument('--save_loc', '-s', default=None, type=str, help="Directory under which to save model and "
                                                                         "statistics.")
    parser.add_argument('--model_save_name', default="FC512Model.pt", type=str, help="Filename under which to save "
                                                                                     "model.")
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--lr', default=0.0001, type=float, help="Learning rate to use for training. Note that "
                                                                 "lr=0.0001 is a common choice.")

    args = parser.parse_args()

    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'basic': {
                'format': '%(message)s',
            },
            'detailed': {
                'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'basic',
                'level': 'WARNING',
            }
        },
        'loggers': {
            'trojai_rl': {
                'handlers': ['console'],
            },
        },
        'root': {
            'level': 'INFO',
        },
    })

    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'

    train_env_factory = RAMEnvFactory()
    test_env_factory = RAMEnvFactory()

    clean_train_args = dict()
    triggered_train_args = dict(poison='add_100', poison_behavior=args.poison_behavior)
    # for testing, by default, insert trigger but don't alter reward function, but allow alternate behaviors
    poison_test_args = dict(poison='add_100', poison_behavior=args.test_poison_behavior)

    train_env_cfgs = [WrappedBoxingConfig(**clean_train_args) for _ in range(args.num_clean_train)] + \
                     [WrappedBoxingConfig(**triggered_train_args) for _ in range(args.num_triggered_train)]
    intermediate_test_cfgs = [TestConfig(WrappedBoxingConfig(**clean_train_args),
                                         count=args.int_num_clean_test),
                              TestConfig(WrappedBoxingConfig(**poison_test_args),
                                         count=args.int_num_triggered_test)]
    test_cfgs = [TestConfig(WrappedBoxingConfig(**clean_train_args),
                            count=args.num_clean_test),
                 TestConfig(WrappedBoxingConfig(**poison_test_args),
                            count=args.num_triggered_test)]
    if args.use_argmax_action_test:
        test_cfgs += [TestConfig(WrappedBoxingConfig(**clean_train_args),
                                 count=args.num_clean_test,
                                 agent_argmax_action=True),
                      TestConfig(WrappedBoxingConfig(**poison_test_args),
                                 count=args.num_triggered_test,
                                 agent_argmax_action=True)]

    env = BoxingRAMObsWrapper(WrappedBoxing(WrappedBoxingConfig(**clean_train_args)))
    model = FC512Model(env.observation_space, env.action_space)
    model.to(device)

    # set up optimizer
    optimizer_cfg = TorchACOptConfig(train_env_cfgs=train_env_cfgs,
                                     test_cfgs=test_cfgs,
                                     algorithm='ppo',
                                     num_frames=args.num_frames,
                                     num_frames_per_proc=args.num_frames_per_proc,
                                     epochs=args.num_epochs,
                                     test_freq_frames=args.test_freq_frames,
                                     test_max_steps=args.test_max_steps,
                                     learning_rate=args.lr,
                                     value_loss_coef=1.0,
                                     clip_eps=0.1,
                                     device=device,
                                     intermediate_test_cfgs=intermediate_test_cfgs,
                                     instantiate_env_in_worker=False,
                                     eval_stats=eval_stats,
                                     aggregate_test_results=aggregate_results,
                                     early_stop=early_stop,
                                     preprocess_obss=model.preprocess_obss)
    optimizer = TorchACOptimizer(optimizer_cfg)

    # turn arguments into a dictionary that we can save as run information
    save_info = vars(args)

    # set data saving parameters
    if args.save_loc:
        save_path = args.save_loc
    else:
        save_path = ''
        save_info['save_loc'] = os.path.abspath('')

    # set up runner and create model
    runner_cfg = RunnerConfig(train_env_factory, test_env_factory, model, optimizer,
                              model_save_dir=os.path.join(save_path, 'models/'),
                              stats_save_dir=os.path.join(save_path, 'stats/'),
                              filename=args.model_save_name,
                              save_info=save_info)
    runner = Runner(runner_cfg)
    runner.run()
