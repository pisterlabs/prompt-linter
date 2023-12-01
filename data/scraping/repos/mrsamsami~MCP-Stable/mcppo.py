import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
import stable_baselines3
import numpy as np
from tqdm.auto import tqdm

class DirAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, direction=0):
        """
        direction: angle in degrees, between 0 and 360 used to specify the desired heading of the agent. Measured anti-clockwise
        """
        assert 0 <= direction <= 360
        self.direction = direction
        direction = direction / 180 * np.pi
        self.desired_heading = np.round((np.cos(direction), np.sin(direction)), 3)

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)


    def step(self, a):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(a, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = np.dot(xy_velocity, self.desired_heading)

        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

import gym

class GoalAnt(DirAntEnv):
    def __init__(self, direction=0):
        super(GoalAnt, self).__init__(direction)
        # NOTE TO SELF: MujocoEnv calls env.step and uses the returned obs to set the observation_space, hence there is no need to manually change the observation space here.

    def step(self, a):
        obs, reward, done, info = super().step(a)
        obs = np.concatenate((obs, self.desired_heading), 0)
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        obs = np.concatenate((obs, self.desired_heading), 0)
        return obs

class RandomGoalAnt(DirAntEnv):
    def __init__(self, direction=0, direction_range=(0, 270), direction_list=None):
        self.direction_range = direction_range
        self.direction_list = direction_list
        super(RandomGoalAnt, self).__init__(direction)
        # NOTE TO SELF: MujocoEnv calls env.step and uses the returned obs to set the observation_space, hence there is no need to manually change the observation space here.

    def step(self, a):
        obs, reward, done, info = super().step(a)
        obs = np.concatenate((obs, self.desired_heading), 0)
        return obs, reward, done, info

    def reset(self):
        # self.direction = np.random.randint(self.direction_range[0], self.direction_range[1])
        self.direction = np.random.choice([0, 90, 180, 270, 45, 135, 225, 315])
        direction = self.direction / 180 * np.pi
        self.desired_heading = np.round((np.cos(direction), np.sin(direction)), 3)
        obs = super().reset()
        obs = np.concatenate((obs, self.desired_heading), 0)
        return obs

import logging
from gym.envs.registration import register

register(
    id='NewAnt-v2',
    entry_point=DirAntEnv,
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='NewGoalAnt-v2',
    entry_point=GoalAnt,
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='NewRandomGoalAnt-v2',
    entry_point=RandomGoalAnt,
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

import torch
from torch import nn

def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    return model


from torch.functional import Tensor
# TODO:
"""
forward actor should return just mean and log_std, not the distribution itself. Also IMPORTANT, it should return the log_std not std itself which is what we are returning rn. Also value_net should return penultimate layer output. Since they do a final linear layer in the code base itself.
Another important factor, the forward_actor that is here is used to produce the embedding that is then transformed through a final action_net. We do not need an action net since that will mess with the distributions that we produce. Ill need to overwrite it then.
"""
'''Code adopted from: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#advanced-example'''
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch
import torch as th
from torch import nn, distributions
from functools import partial

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution, Distribution
from stable_baselines3.common.type_aliases import Schedule
from torch.distributions import Normal

direction = 0
model_selector = np.zeros(4)
model_selector[direction] = 1

class MCPPOHiddenLayers(nn.Module):
    """
    Custom hidden network architecture for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param state_dim: dimension of the input features
    """
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        models: List[nn.Module],
        learn_log_std: bool,
    ):
        super(MCPPOHiddenLayers, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions

        self.latent_dim_pi = 512
        self.latent_dim_vf = 64

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.num_primitives = len(models)
        self.action_dim = action_dim
        self.learn_log_std = learn_log_std

        # build the Policy network hidden layers

        # Gating Function:
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_primitives),
            nn.Sigmoid()
        )

        self.primitive_state_encoder = nn.Identity()

        self.primitives =  [freeze_model(mod) for mod in models]

        # build the Value network hidden layers
        self.value_net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, self.latent_dim_vf),
            nn.ReLU(),
            nn.Linear(self.latent_dim_vf, self.latent_dim_vf),
            nn.ReLU(),
        )


    def forward_weights(self,features: th.Tensor) -> th.Tensor:
        state, goal = torch.split(features, [self.state_dim, self.goal_dim], -1)
        state_embed = self.state_encoder(state)
        goal_embed = self.goal_encoder(goal)
        embed = th.cat((state_embed, goal_embed), -1)
        weights = self.gate(embed)
        return weights

    def forward_primitive(self, i: int, state: th.Tensor) -> List[th.Tensor]:
        model = self.primitives[i]
        get_action = lambda embed: model.action_net(model.mlp_extractor.forward_actor(embed))
        prim_embed = self.primitive_state_encoder(state)
        mu = get_action(prim_embed)
        log_std = self.primitives[i].log_std.clone()
        sigma = th.ones_like(mu) * log_std.exp()
        return mu, sigma

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        bs = features.shape[0]

        state, _ = torch.split(features, [self.state_dim, self.goal_dim], -1)
        weights = self.forward_weights(features)

        mus, sigmas = [], []
        i=0
        mu, sigma = self.forward_primitive(i, state)
        mus.append(mu)
        sigmas.append(sigma)

        i=1
        mu, sigma = self.forward_primitive(i, state)
        mus.append(mu)
        sigmas.append(sigma)

        i=2
        mu, sigma = self.forward_primitive(i, state)
        mus.append(mu)
        sigmas.append(sigma)

        i=3
        mu, sigma = self.forward_primitive(i, state)
        mus.append(mu)
        sigmas.append(sigma)

        mus = torch.stack(mus, 1)
        sigmas = torch.stack(sigmas, 1)
        weights = weights[..., None]

        assert mus.shape[0] == bs and mus.shape[1] == self.num_primitives and mus.shape[2] == self.action_dim
        assert sigmas.shape[0] == bs and sigmas.shape[1] == self.num_primitives and sigmas.shape[2] == self.action_dim

        denom = ( weights / sigmas).sum(-2)
        unnorm_mu = (weights / sigmas * mus).sum(-2)

        mean = unnorm_mu / denom
        assert mean.shape == (bs, self.action_dim)
        if not self.learn_log_std:
            scale_tril = 1 / denom
            return mean, scale_tril
        else:
            return mean


    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        value = self.value_net(features)
        return value


    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)


class MPPO(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        state_dim: int = 11,
        goal_dim: int = 2,
        models: List[nn.Module] = None,
        learn_log_std: bool = True,
        *args,
        **kwargs,
    ):

        assert state_dim + goal_dim == observation_space.shape[0]
        self.mcppo_state_dim = state_dim
        self.mcppo_goal_dim = goal_dim
        self.models = models
        self.learn_log_std = learn_log_std

        super(MPPO, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MCPPOHiddenLayers(self.mcppo_state_dim, self.mcppo_goal_dim, int(np.prod(self.action_space.shape)), self.models, self.learn_log_std)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            if self.learn_log_std:
                self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                    latent_dim=latent_dim_pi, log_std_init=self.log_std_init
                )
                self.action_net = nn.Identity()
            else:
                self.action_net = nn.Identity()
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

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        assert isinstance(observation, th.Tensor)
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        if not self.learn_log_std:
            latent_pi, latent_std = latent_pi
        assert isinstance(latent_pi, th.Tensor)
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            if self.learn_log_std:
                return self.action_dist.proba_distribution(mean_actions, self.log_std)
            else:
                self.action_dist.distribution = Normal(mean_actions, latent_std)
                return self.action_dist
        else:
            raise ValueError("Invalid action distribution")

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def predict_weights(self, observation: np.ndarray) -> np.ndarray:
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            features = self.extract_features(observation)
            weights = self.mlp_extractor.forward_weights(features)
        # Convert to numpy
        weights = weights.cpu().numpy()

        # Remove batch dimension if needed
        if not vectorized_env:
            weights = weights[0]

        return weights

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

def encode_gif(frames, fps):
    from subprocess import PIPE, Popen

    h, w, c = frames[0].shape
    pxfmt = {1: "gray", 3: "rgb24"}[c]
    cmd = " ".join(
        [
            "ffmpeg -y -f rawvideo -vcodec rawvideo",
            f"-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex",
            "[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse",
            f"-r {fps:.02f} -f gif -",
        ]
    )
    proc = Popen(cmd.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError("\n".join([" ".join(cmd), err.decode("utf8")]))
    del proc
    return out


def write_gif_to_disk(frames, filename, fps=10):
    """
    frame: np.array of shape TxHxWxC
    """
    try:
        frames = encode_gif(frames, fps)
        with open(filename, "wb") as f:
            f.write(frames)
        tqdm.write(f"GIF saved to {filename}")
    except Exception as e:
        tqdm.write(frames.shape)
        tqdm.write("GIF Saving failed.", e)

class SaveVideoCallback(BaseCallback):
    """
    Callback for saving the setpoint tracking plot(the check is done every ``eval_freq`` steps)
    :param eval_env: (gym.Env) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, eval_env, eval_freq=10000, video_freq=10000, vec_normalise=False, log_dir=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        assert isinstance(eval_freq, int)
        assert isinstance(video_freq, int)
        assert eval_freq <= video_freq and video_freq % eval_freq == 0
        self.eval_freq = eval_freq
        self.video_freq = video_freq
        self.save_path = None
        self.vec_normalise = vec_normalise
        if log_dir is not None:
            self.log_dir = log_dir
            self.save_path = os.path.join(log_dir, "images")

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def preprocess(self, obs):
        if self.vec_normalise:
            return self.model.env.normalize_obs(obs)
        else:
            return obs

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            obs = self.eval_env.reset()
            if self.n_calls % self.video_freq == 0:
                img = self.eval_env.render("rgb_array")
                imgs = [img]
            done = False
            tot_r = 0.0
            weights=[]
            pbar = tqdm(total=1000)
            print(f"Begin Evaluation")
            while not done:
                action, _ = self.model.predict(self.preprocess(obs), deterministic=True)
                weight = self.model.policy.predict_weights(obs)
                weights.append(weight)
                obs, reward, done, info = self.eval_env.step(action)
                if self.n_calls % self.video_freq == 0:
                    img = self.eval_env.render("rgb_array")
                    imgs.append(img)
                tot_r += reward
                pbar.update(1)
            pbar.close()
            print(f"Evaluation Reward: {tot_r}")
            weights = np.array(weights).squeeze(1)
            fname=os.path.join(self.save_path, "weights.npy")
            np.save(fname, weights)
            ep_len = weights.shape[0]
            print(f"Ep Len: {ep_len}")
            for i in range(weights.shape[1]):
                plt.plot(weights[:, i], label=f"Model {i}")
            plt.xlim(0, ep_len)
            plt.ylim(0, 1)
            # plt.title(f"Weights assigned to PPO primitives {self.eval_env.envs[0].direction}")
            plt.title(f"Weights assigned to PPO primitives")
            plt.tight_layout()
            plt.legend()
            fname=os.path.join(self.save_path, "weights.jpg")
            plt.savefig(fname, bbox_inches="tight", dpi=120)
            plt.close()
            if self.save_path is not None and self.n_calls % self.video_freq == 0:
                imgs = np.array(imgs)
                fname=os.path.join(self.save_path, "eval_video.gif")
                fps = 30 if ep_len < 200 else 60
                write_gif_to_disk(imgs, fname, fps)

        return True

import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecCheckNan, VecNormalize
import torch
import copy

env_model="NewGoalAnt-v2"
env_model="NewRandomGoalAnt-v2"

learn_log_std=False

run_id = f"transfer_mcppo"
direction = direction
algo = "PPO"
logdir = "logs"
seed = 0
vec_normalise = False

num_envs = 4
training_timesteps = int(3e6)
checkpoint_freq = 200000
eval_freq = 50000
video_freq = 100000

torch.autograd.set_detect_anomaly(True)

print("Algorithm: ", algo)
tag_name = os.path.join(f"{env_model}", f"{algo}_{run_id}")
print("Run Name: ", tag_name)

log_dir = os.path.join(logdir, tag_name, f"seed{str(seed)}")
model_dir = os.path.join(log_dir, "models")
tbdir = os.path.join(log_dir, "tb_logs")
mon_dir = os.path.join(log_dir, "gym")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(mon_dir, exist_ok=True)

env_direction = {
    0: 0,
    1: 180,
    2: 90,
    3: 270,
}

env_kwargs={
    'direction': env_direction[direction],
    'direction_range': [270, 360],
    }

assert "GoalAnt-v2" in env_model
env = make_vec_env(env_model, n_envs=num_envs, monitor_dir=mon_dir, env_kwargs=env_kwargs, seed=seed)

assert not vec_normalise
if vec_normalise:
    if os.path.exists(os.path.join(log_dir, "vec_normalize.pkl")):
        print("Found VecNormalize Stats. Using stats")
        env = VecNormalize.load(os.path.join(log_dir, "vec_normalize.pkl"), env)
    else:
        print("No previous stats found. Using new VecNormalize instance.")
        env = VecNormalize(env)
else:
    print("Not using VecNormalize")

env = VecCheckNan(env, raise_exception=True)

checkpoint_callback = CheckpointCallback(int(checkpoint_freq // num_envs), model_dir, tag_name, 2)
eval_env = make_vec_env(env_model, n_envs=1, monitor_dir=mon_dir, env_kwargs=env_kwargs)
save_video_callback = SaveVideoCallback(eval_env, int(eval_freq // num_envs), int(video_freq // num_envs), vec_normalise, log_dir, 2)
callback = CallbackList([checkpoint_callback, save_video_callback])

custom_objects = {
    "lr_schedule": lambda x: .003,
    "clip_range": lambda x: .02
}

models = [PPO.load(f'logs/NewAnt-v2/PPO_vec_norm_False_direction_{dir}_5M/seed0/final.zip', custom_objects=custom_objects) for dir in range(4)]
models = [mod.policy for mod in models]

policy_kwargs={
    "state_dim": env.observation_space.shape[0] - 2,
    "goal_dim": 2,
    "models": copy.deepcopy(models),
}
mppo_model = PPO(MPPO, env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=tbdir, seed=seed)

mppo_model.learn(total_timesteps=training_timesteps, callback=callback)

checkpoint_dir = os.path.join(log_dir, "final")
mppo_model.save(checkpoint_dir)