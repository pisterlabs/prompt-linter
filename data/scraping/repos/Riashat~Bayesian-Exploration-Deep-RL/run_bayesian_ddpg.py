from ddpg_bayesian_thompson import DDPG as DDPG_Thompson
from ddpg_bayesian_mean import DDPG as DDPG_Mean
from ddpg_bayesian import DDPG as DDPG_Bayesian
from dropout_exploration import MCDropout
from deterministic_mlp_policy_bayesian import DeterministicMLPPolicy
from continuous_mlp_q_function_bayesian import ContinuousMLPQFunction
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
from rllab.misc import ext
import pickle
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("type", help="Type of DDPG to run: unified, unified-gated, regular")
parser.add_argument("env", help="The environment name from OpenAIGym environments")
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--plot", action="store_true")
# parser.add_argument("--data_dir", default="./data/")
args = parser.parse_args()

stub(globals())
ext.set_seed(1)

supported_gym_envs = ["MountainCarContinuous-v0", "Hopper-v1", "Walker2d-v1", "Humanoid-v1", "Reacher-v1", "HalfCheetah-v1", "Swimmer-v1", "HumanoidStandup-v1"]

other_env_class_map  = { "Cartpole" :  CartpoleEnv}

if args.env in supported_gym_envs:
    gymenv = GymEnv(args.env, force_reset=True, record_video=False, record_log=False)
    # gymenv.env.seed(1)
else:
    gymenv = other_env_class_map[args.env]()


env = TfEnv(normalize(gymenv))

policy = DeterministicMLPPolicy(
    env_spec=env.spec,
    name="policy",
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(100, 50, 25),
    hidden_nonlinearity=tf.nn.relu,
)

es = MCDropout(env_spec=env.spec)

qf = ContinuousMLPQFunction(env_spec=env.spec,
                            hidden_sizes=(100,100),
                            hidden_nonlinearity=tf.nn.relu,)


ddpg_type_map = {"Thompson" : DDPG_Thompson, "Mean" : DDPG_Mean, "Bayesian" : DDPG_Bayesian}

ddpg_class = ddpg_type_map[args.type]

## loops:
num_experiments = 1
batch_size_values = [64]




for b in range(len(batch_size_values)): 
    
    for e in range(num_experiments):

        algo = ddpg_class(
            env=env,
            policy=policy,
            es=es,
            qf=qf,
            batch_size=64,
            max_path_length=env.horizon,
            epoch_length=1000,
            min_pool_size=10000,
            n_epochs=args.num_epochs,
            discount=0.99,
            scale_reward=1.0,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            plot=args.plot,
        )


        run_experiment_lite(
            algo.train(),
            # log_dir=args.data_dir,
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            exp_name="Trial_Bayesian_Exploration/",
            seed=1,
            plot=args.plot,
        )
