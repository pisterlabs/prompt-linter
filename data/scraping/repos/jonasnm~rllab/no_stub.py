import argparse

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn (pip install seaborn) for better plotting!')

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("env", help="The environment name from OpenAIGym environments")
parser.add_argument("--algorithm", default='VPG', help='RL algorithm(0-3): VPG(0), TRPO(1), TNPG(2) or DDPG(3)')
parser.add_argument("--reward", default='absolute', help="The reward function from OpenAIGym diabetes environment")
parser.add_argument("--n_itr", default=200, type=int)
parser.add_argument("--data_dir", default="./data_trpo/")

args = parser.parse_args()


if args.algorithm == 0:
    from rllab.algos.vpg import VPG
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
elif args.algorithm == 1:
    from rllab.algos.vpg import TRPO
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
elif args.algorithm == 2:
    from rllab.algos.vpg import TNPG
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# elif args.algorithm == 3:
    # from rllab.algos.vpg import DDPG
    # from rllab.exploration_strategies.ou_strategy import OUStrategy
    # from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
    # from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

# ==========================================================================
# OpenAI diabetes envs - HovorkaInterval starts at the same value every time,
# HovorkaIntervalRandom starts at a random value
# ==========================================================================

def run_task(*_):
    env = normalize(GymEnv(args.env))
    # env.wrapped_env.env.env.env.reward_flag = 'absolute'
    env.wrapped_env.env.env.reward_flag = args.reward


    baseline = LinearFeatureBaseline(env_spec=env.spec)

    learn_std = True
    init_std=2

    # hidden_sizes=(8,)
    hidden_sizes=(32, 32)
    # hidden_sizes=(100, 50, 25)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=hidden_sizes,
        learn_std=learn_std,
        init_std=init_std
    )

    # =======================
    # Defining the algorithm
    # =======================
    batch_size = 5000
    n_itr = args.n_itr
    gamma = .9
    step_size = 0.01

    if args.algorithm == 0:
        algo = VPG(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            n_itr=n_itr,
            discount=gamma,
            step_size=step_size
        )
    if args.algorithm == 1:
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            n_itr=n_itr,
            discount=gamma,
            step_size=step_size
        )
    if args.algorithm == 2:
        algo = TNPG(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            n_itr=n_itr,
            discount=gamma,
            step_size=step_size
        )
    # if args.algorithm == 4:
        # algo = DDPG(
            # env=env,
            # policy=policy,
            # baseline=baseline,
            # batch_size=batch_size,
            # n_itr=n_itr,
            # discount=gamma,
            # step_size=step_size
        # )
    algo.train()

    return algo


# Running and saving the experiment
run_experiment_lite(
    run_task,
    # algo.train(),
    log_dir=args.data_dir,
    # n_parallel=2,
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    # exp_prefix="Reinforce_" + env_name,
    # exp_prefix=data_dir
    plot=False
)


