# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces

cv2.ocl.setUseOpenCL(False)

# ATARI Warppers from openai-baseline
# Not sure same as offline data collect's NatureDQN's wrapper
# TODO Check
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=((shp[0] * k,) + shp[1:]), dtype=env.observation_space.dtype
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


def wrap_atari(env, max_episode_steps=None):
    assert "NoFrameskip" in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    assert max_episode_steps is None

    return env


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))


def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari."""
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = ImageToPyTorch(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import argparse
import os
import random
import re
import time
from distutils.util import strtobool

import gym
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.spaces import Discrete
# from gym.wrappers import Monitor
from torch.utils.data.dataset import IterableDataset
from torch.utils.tensorboard import SummaryWriter
import d4rl_atari

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN agent")
    # Common arguments
    parser.add_argument(
        "--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment"
    )
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4", help="the id of the environment")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=2, help="seed of the experiment")
    parser.add_argument("--iters", type=int, default=200, help="total iters of experiments")
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will not be enabled by default",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="run the script in production mode and use wandb to log outputs",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)",
    )
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    # parser.add_argument(
    #     "--offline-dataset-id", type=str, default="expert-v0", help="the id of the offline dataset gym environment"
    # )
    parser.add_argument("--buffer-size", type=int, default=1000000, help="the replay memory buffer size")
    parser.add_argument("--buffer-num", type=int, default=5, help="the replay buffer loaded num")
    parser.add_argument("--num-heads", type=int, default=200, help="the head num of rem dqn network")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument(
        "--target-network-frequency", type=int, default=8000, help="the timesteps it takes to update the target network"
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--batch-size", type=int, default=32, help="the batch size of sample from the reply memory")
    parser.add_argument("--save-frequency", type=int, default=5, help="the epoch it takes to save parameters")
    parser.add_argument("--eval-steps", type=int, default=125000, help="the eval steps for eval after each epoch")
    parser.add_argument("--train-frequency", type=int, default=4, help="the frequency of training")
    parser.add_argument("--train-step", type=int, default=250000, help="the train step per iter")
    
    # args = parser.parse_args()
    args = parser.parse_args('--seed 1 --env-id PongNoFrameskip-v4 --eval-step 30000 --buffer-num 1'.split(' '))
    print('args.batch_size:', args.batch_size)
    if not args.seed:
        args.seed = int(time.time())
    # create offline gym id: 'BeamRiderNoFrameskip-v4' -> 'beam-rider-expert-v0'


class QValueVisualizationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.reset()
        self.image_shape = self.env.render(mode="rgb_array").shape
        self.q_values = [[0.0, 0.0, 0.0, 0.0]]
        # self.metadata['video.frames_per_second'] = 60

    def set_q_values(self, q_values):
        self.q_values = q_values

    def render(self, mode="human"):
        if mode == "rgb_array":
            env_rgb_array = super().render(mode)
            fig, ax = plt.subplots(
                figsize=(self.image_shape[1] / 100, self.image_shape[0] / 100), constrained_layout=True, dpi=100
            )
            df = pd.DataFrame(np.array(self.q_values).T)
            sns.barplot(x=df.index, y=0, data=df, ax=ax)
            ax.set(xlabel="actions", ylabel="q-values")
            fig.canvas.draw()
            X = np.array(fig.canvas.renderer.buffer_rgba())
            Image.fromarray(X)
            # Image.fromarray(X)
            rgb_image = np.array(Image.fromarray(X).convert("RGB"))
            plt.close(fig)
            q_value_rgb_array = rgb_image
            return np.append(env_rgb_array, q_value_rgb_array, axis=1)
        else:
            super().render(mode)


# TRY NOT TO MODIFY: setup the environment
cur_time = time.localtime(time.time())
cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", cur_time)
experiment_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{cur_time}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text(
    "hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
)
if args.track:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=experiment_name,
        monitor_gym=True,
        save_code=True,
    )
    writer = SummaryWriter(f"/tmp/{experiment_name}")
save_path = os.path.join(f"runs/{experiment_name}", 'model.pth')

# TRY NOT TO MODIFY: seeding
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
env = gym.make(args.env_id)
env = wrap_atari(env)
env = gym.wrappers.RecordEpisodeStatistics(env)  # records episode reward in `info['episode']['r']`
# if args.capture_video:
#     env = QValueVisualizationWrapper(env)
#     env = Monitor(env, f"videos/{experiment_name}")
env = wrap_deepmind(
    env,
    episode_life=False,
    clip_rewards=False,
    frame_stack=True,
    scale=False,
)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
# respect the default timelimit
assert isinstance(env.action_space, Discrete), "only discrete action space is supported"


# ALGO LOGIC: initialize agent here:
# tricks taken from https://github.com/cpnota/autonomous-learning-library/blob/6d1111afce0d1582de463326f7d078a86e850551/all/presets/atari/models/__init__.py#L16
# apparently matters
class Linear0(nn.Linear):
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class MultiHeadQNetwork(nn.Module):
    def __init__(self, env, frames=4, num_heads=1, transform_strategy='STOCHASTIC', transform_matrix=None):
        self.num_heads = num_heads
        self.transform_strategy = transform_strategy
        self.transform_matrix = transform_matrix
        super(MultiHeadQNetwork, self).__init__()
        self.network = nn.Sequential(
            Scale(1 / 255),
            nn.Conv2d(frames, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            Linear0(512, env.action_space.n * num_heads),
        )

    def forward(self, x, device):
        # x = torch.Tensor(x).to(device)
        unordered_q_heads = self.network(x)
        unordered_q_heads = torch.reshape(unordered_q_heads, [-1, env.action_space.n, self.num_heads])
        q_heads, q_values = combine_q_functions(unordered_q_heads, self.transform_strategy, self.transform_matrix)
        # q_heads_shape: batch_size, aciton_dim
        # num_cols (def in random_stochastic_matrix, means different ways of q_head combination )
        # we all set it to one
        # in origin REM, when calculate loss, need reduce twice, onces for batch_dim, once for col_dim
        # q_values_shape: batch_size, aciton_dim,
        return q_heads, q_values

def combine_q_functions(q_functions, transform_strategy, transform_matrix=None):
    q_values = torch.mean(q_functions, axis=-1)
    if transform_strategy=='STOCHASTIC':
        # q_functions input shape: (batch_size, num_actions, num_heads)
        # left_stochastic_matrix shape: (num_heads, num_convex_combinations=1(we fix it as 1))
        # q_functions output shape: (batch_size, num_actions, 1)
        # squeeze to (batch_size, num_actions, 1)
        q_functions = torch.matmul(q_functions, transform_matrix).squeeze()
    elif transform_strategy=='IDENTITY':
        pass
    else:
        raise ValueError(
            '{} is not a valid reordering strategy'.format(transform_strategy))
    return q_functions, q_values

def random_stochastic_matrix(dim, dtype=torch.float32, device=device):
    """Generates a random left stochastic matrix."""
    # check dopamine's notebook 
    # after test this has same result
    mat_shape = (dim, 1)
    mat = torch.rand(mat_shape, dtype=dtype, device=device)
    mat /= torch.norm(mat, p=1, dim=0, keepdim=True)
    return mat

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class ExperienceReplayDataset(IterableDataset):
    def __init__(self, env_id, offline_dataset_id):
        offline_env_id = (
        re.sub(r"(?<!^)(?=[A-Z])", "-", env_id).lower().replace("no-frameskip-v4", "") + offline_dataset_id
        )
        self.dataset_env = gym.make(offline_env_id, stack=True)
        self.dataset = self.dataset_env.get_dataset()

    def __iter__(self):
        while True:
            idx = np.random.choice(len(self.dataset["observations"]) - 1)
            yield self.dataset["observations"][idx], self.dataset["actions"][idx], self.dataset["rewards"][idx], self.dataset[
                "observations"
            ][idx + 1], self.dataset["terminals"][idx]
    
    def __len__(self):
        return len(self.dataset['observations'])

transform_matrix = random_stochastic_matrix(dim=args.num_heads)
q_network = MultiHeadQNetwork(env, num_heads=args.num_heads, transform_matrix=transform_matrix).to(device)
target_network = MultiHeadQNetwork(env, num_heads=args.num_heads, transform_matrix=transform_matrix).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
# TODO use Huber loss as in orign REM code
# current torch version is old, no Huber loss
# loss_fn = nn.MSELoss()
loss_fn = nn.SmoothL1Loss()
print(device.__repr__())
print(q_network)

global_train_step = 0
for iteration in range(args.iters):
    
    # Save network and optimizer
    if iteration % args.save_frequency == 0:
        torch.save({
                'iteration': iteration,
                'q_network_state_dict': q_network.state_dict(),
                'target_network_state_dict': target_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)
        
    # ALGO LOGIC: Evaluation every epoch
    # TRY NOT TO MODIFY: start the game
    obs = env.reset()
    done = False
    epsilon = 0.001
    episode_returns = []
    episode_lengths = []
    eval_time = 0
    for eval_step in range(args.eval_steps):
        obs = np.array(obs)
        obs = torch.Tensor(obs).to(device)
        logits = q_network.forward(obs.reshape((1,) + obs.shape), device)[1]
        if args.capture_video:
            env.set_q_values(logits.tolist())
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(logits, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = env.step(action)

        if "episode" in info.keys():
            print(f"iter={iteration}, eval_step={eval_step}, episodic_return={info['episode']['r']}")
            episode_returns.append(info["episode"]["r"])
            episode_lengths.append(info["episode"]["l"])
            eval_time += info["episode"]["t"]
    
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        if done:
            obs = env.reset()
            
    # TRY NOT TO MODIFY: record rewards for plotting purposes
    print("iter={}, episodic_return={:.2f}".format(iteration, np.mean(episode_returns)))
    writer.add_scalar("Charts/episode_return_mean", np.mean(episode_returns), iteration)         
    writer.add_scalar("Charts/episode_return_var", np.var(episode_returns), iteration)
    writer.add_scalar("Charts/episode_num", len(episode_returns), iteration)  
    writer.add_scalar("Charts/episode_length_mean", np.mean(episode_lengths), iteration)       
    writer.add_scalar("Charts/eval_time", eval_time, iteration)   
    
    buffer_num = args.buffer_num
    data_loaders = []
    dataset_ids =  np.random.choice(np.arange(1, 51), buffer_num, replace=False)
    for dataset_id in dataset_ids:
        dataset = ExperienceReplayDataset(args.env_id, 'epoch-' + str(dataset_id) + '-v0')
        data_loader = iter(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=2))
        data_loaders.append(data_loader)
    # for batch_num in range(int(np.ceil(len(dataset) / args.batch_size))):
    for train_step in range(args.train_step):
        if train_step % args.train_frequency == 0:
            # sample batch
            data_loader = data_loaders[np.random.randint(0, buffer_num)]
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = next(data_loader)
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = (
                s_obs.to(device),
                s_actions.to(device),
                s_rewards.to(device),
                s_next_obses.to(device),
                s_dones.to(device),
            )
            
            # REM change combine matrix
            transform_matrix = random_stochastic_matrix(dim=args.num_heads)
            q_network.transform_matrix = transform_matrix
            target_network.transform_matrix = transform_matrix
            
            with torch.no_grad():
                target_q = target_network.forward(s_next_obses, device)[0] # return q_heads, q_values, trainning use q_heads
                target_max = torch.max(target_q, dim=1)[0] # torch.max return (values, indices), we use values
                td_target = s_rewards + args.gamma * target_max * (1 - s_dones)
            old_val = q_network.forward(s_obs, device)[0].gather(1, s_actions.long().view(-1, 1)).squeeze()
            loss = loss_fn(td_target, old_val)
            
            if global_train_step % 1000 == 0:
                writer.add_scalar("losses/td_loss", loss, global_train_step)
                writer.add_scalar("losses/average_q_values", old_val.mean().item(), global_train_step)
                writer.add_scalar("losses/average_combined_target_values", target_q.mean().item(), global_train_step)
        
            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
            optimizer.step()

            # update the target network
            if train_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
            
        global_train_step += 1

env.close()
writer.close()
