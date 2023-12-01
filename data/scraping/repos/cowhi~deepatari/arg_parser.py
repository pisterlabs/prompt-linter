import argparse
from deepatari import __version__

def str2bool(v):
    """ Helps to avoid confusion for truth values. """
    return v.lower() in ("yes", "true", "t", "1")

def parse_args(args):
    """ Parse command line parameters.

    Args:
        args (tuple[str]): All settings either default or set via command line arguments.

    Returns:
        args (argparse.Namespace): All settings either default or set via command line arguments.

    """
    parser = argparse.ArgumentParser(
            description="Framework to facilitate learning in the Atari game playing domain.")
    parser.add_argument(
            '-v',
            '--version',
            action='version',
            version='deepatari {ver}'.format(ver=__version__))

    exparg = parser.add_argument_group('Experiment')
    exparg.add_argument("--exp_type", default="AtariExp", help="Choose experiment implementation.")
    exparg.add_argument("--env_type", default="AtariEnv", help="Choose environment implementation.")
    exparg.add_argument("--agent_type", default="AtariAgent", help="Choose agent implementation.")
    exparg.add_argument("--memory_type", default="ReplayMemory", help="Choose memory implementation.")
    exparg.add_argument('--with', dest="learner_type", default='DQNNeon', help='Choose network implementation.')
    exparg.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")
    exparg.add_argument("--log_type", choices=["file", "stdout"], default="file", help="Where to log to.")
    exparg.add_argument("--log_stats", type=str2bool, default=True, help="Turn stats on and off.")
    exparg.add_argument("--epochs", type=int, default=200, help="How many epochs to run.")
    exparg.add_argument("--train_steps", type=int, default=250000, help="How many training steps per epoch.")
    exparg.add_argument("--test_steps", type=int, default=125000, help="How many testing steps after each epoch.")
    exparg.add_argument("--random_seed", type=int, default=666, help="Random seed for repeatable experiments.")
    exparg.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu', help='Choose the backend type, which will perform the calculations.')
    exparg.add_argument('--device_id', type=int, default=0, help='gpu device id (only used with GPU backend)')
    exparg.add_argument('--datatype', choices=['float16', 'float32', 'float64'], default='float32', help='default floating point precision for backend [f64 for cpu only]')

    envarg = parser.add_argument_group('Environment')
    envarg.add_argument(
            "--game",
            default='Breakout-v0',
            type=str, action='store',
            help='ROM or environment ID from OpenAI gym to run (default: %(default)s)')
    envarg.add_argument("--display_screen", type=str2bool, default=False, help="Display game screen during training and testing.")
    envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
    envarg.add_argument("--repeat_action_probability", type=float, default=0.0, help="Probability, that chosen action will be repeated. Otherwise random action is chosen during repeating.")
    envarg.add_argument("--color_averaging", type=str2bool, default=True, help="Perform color averaging with previous frame.")
    envarg.add_argument("--frame_width", type=int, default=84, help="Frame width after resize.")
    envarg.add_argument("--frame_height", type=int, default=84, help="Frame height after resize.")
    envarg.add_argument("--sequence_length", type=int, default=4, help="How many image frames form a state.")

    memarg = parser.add_argument_group('Replay memory')
    memarg.add_argument("--memory_size", type=int, default=1000000, help="Maximum size of replay memory.")
    memarg.add_argument("--fill_mem_size", type=int, default=50000, help="Populate replay memory with fill_mem_size random steps before starting to learn.")

    antarg = parser.add_argument_group('Agent')
    antarg.add_argument("--epsilon_start", type=float, default=1, help="Exploration rate (epsilon) at the beginning of decay.")
    antarg.add_argument("--epsilon_end", type=float, default=0.1, help="Exploration rate (epsilon) at the end of decay.")
    antarg.add_argument("--epsilon_decay_steps", type=float, default=1000000, help="How many steps to decay the exploration rate (epsilon) .")
    antarg.add_argument("--epsilon_test", type=float, default=0.05, help="Exploration rate (epsilon) used during testing.")
    antarg.add_argument("--train_frequency", type=int, default=4, help="Perform training after this many game steps.")
    antarg.add_argument("--train_repeat", type=int, default=1, help="Number of times to sample minibatch during training.")
    antarg.add_argument("--random_starts", type=int, default=30, help="Perform max this number of dummy actions after game restart, to produce more random game dynamics.")

    netarg = parser.add_argument_group('Network')
    netarg.add_argument("--train_all", type=str2bool, default=False, help="Use all possible actions or minimum set for training.")
    netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate (alpha).")
    netarg.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate for future rewards (gamma).")
    netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
    netarg.add_argument('--optimizer', choices=['rmsprop', 'adam', 'adadelta', 'sgd'], default='rmsprop', help='Network optimization algorithm.')
    netarg.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp and Adadelta algorithms.")
    netarg.add_argument("--rms_epsilon", type=float, default=1e-6, help="Epsilon for RMSProp")
    netarg.add_argument("--momentum", type=float,  default=0., help="Momentum for RMSProp")
    netarg.add_argument("--clip_error", type=float, default=1., help="Clip error term in update between this number and its negative to avoid gradient become zero.")
    netarg.add_argument("--target_update_frequency", type=int, default=10000, help="Copy weights of training network to target network after this many steps.")
    netarg.add_argument("--min_reward", type=float, default=-1, help="Minimum reward.")
    netarg.add_argument("--max_reward", type=float, default=1, help="Maximum reward.")
    netarg.add_argument("--batch_norm", type=str2bool, default=False, help="Use batch normalization in all layers.")
    netarg.add_argument('--stochastic_round', const=True, type=int, nargs='?', default=False, help="Use stochastic rounding (will round to BITS number of bits if specified).")
    netarg.add_argument("--load_weights", help="Load network from file.")

    #otharg = parser.add_argument_group('Other')
    #otharg.add_argument("--visualization_filters", type=int, default=4, help="Number of filters to visualize from each convolutional layer.")
    #otharg.add_argument("--visualization_file", help="Write layer visualization to this file.")
    #otharg.add_argument("--record_path", help="Record game screens under this path. Subfolder for each game is created.")
    #otharg.add_argument("--record_sound_filename", help="Record game sound in this file.")
    #otharg.add_argument("--play_games", type=int, default=0, help="How many games to play, suppresses training and testing.")

    args = parser.parse_args()
    return args
