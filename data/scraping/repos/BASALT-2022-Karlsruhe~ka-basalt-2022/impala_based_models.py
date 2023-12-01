import os

import torch as th
import torch.nn as nn
from openai_vpt.lib.impala_cnn import ImpalaCNN
from openai_vpt.lib.policy import ImgPreprocessing
from openai_vpt.lib.util import FanInInitReLULayer
from openai_vpt.agent import MineRLAgent
from typing import Dict

import gym
from behavioural_cloning import load_model_parameters


class ImpalaLinear(nn.Module):
    """ImpalaCNN followed by a linear layer.

    :param cnn_outsize: impala output dimension
    :param output_size: output size of the linear layer.
    :param dense_init_norm_kwargs: kwargs for linear FanInInitReLULayer
    :param init_norm_kwargs: kwargs for 2d and 3d conv FanInInitReLULayer
    """

    def __init__(
        self,
        output_size: int,
        cnn_outsize: int = 256,
        cnn_width: int = 1,
    ):
        super().__init__()

        if cnn_width == 1:
            chans=(64, 128, 128)
        elif cnn_width == 2:
            chans=(128, 256, 256)
        elif cnn_width == 3:
            chans=(192, 384, 384)
        else:
            raise ValueError(f"There is no VPT model with width {model_width}!")
        self.cnn_width = cnn_width

        # TODO use img_preprocess?
        self.img_preprocess = ImgPreprocessing(img_statistics=None, scale_img=True)
        self.cnn = ImpalaCNN(
            inshape=[128, 128, 3],
            chans=chans,
            outsize=256,
            nblock=2,
            first_conv_norm=False,
            post_pool_groups=1,
            dense_init_norm_kwargs={"batch_norm": False, "layer_norm": True},
            init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize,
            output_size,
            layer_type="linear",
            **{"batch_norm": False, "layer_norm": True},
        )

    def forward(self, img):
        if len(img.shape[:-3]) == 1:
            # Need to add fictitious time dimension
            img = img.unsqueeze(1)
        elif len(img.shape[:-3]) == 0:
            # Need to add batch and time dimension
            img = img.unsqueeze(0).unsqueeze(0)
        out = self.linear(self.cnn(self.img_preprocess(img)))
        # Remove fictitious time dimension
        out = out.squeeze(1)
        return out

    def load_cnn_weights(self, model_path="data/VPT-models"):
        self.cnn.load_state_dict(th.load(os.path.join(model_path, f"ImpalaCNN-{self.cnn_width}x.weights")))


class ImpalaBinaryClassifier(nn.Module):
    """Classification network based on ImpalaCNN"""

    def __init__(self, cnn_outsize=256, cnn_width=1, model_path="data/VPT-models", hidden_size=256):
        super().__init__()
        self.impala_linear = ImpalaLinear(hidden_size, cnn_outsize, cnn_width)
        self.out_linear = nn.Linear(hidden_size, 2)
        self.impala_linear.load_cnn_weights(model_path)
        self.hidden_size = hidden_size

    def forward(self, obs):
        return self.out_linear(self.impala_linear(obs))

    @th.no_grad()
    def class_probs(self, obs):
        return nn.functional.softmax(self.forward(obs), dim=-1)

    @th.no_grad()
    def predict(self, obs):
        return self.forward(obs).argmax(dim=-1)


class ImpalaRewardModel(nn.Module):
    """Reward model based on ImpalaCNN"""

    def __init__(self, cnn_outsize=256, cnn_width=1, model_path="data/VPT-models"):
        super().__init__()
        self.impala_linear = ImpalaLinear(cnn_outsize, 1, cnn_width)
        self.impala_linear.load_cnn_weights(model_path)
        # TODO normalizing layer
        #self.normalized_reward_layer = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        return self.impala_linear(obs)


def save_impala_cnn_weights(model_width=1):
    """
    Saves ImpalaCNN weights.

    `model_width in [1, 2, 3]` according to the three VPT model sizes
    """
    # Setup a MineRL environment
    minerl_env_str = "MineRLBasaltFindCave"
    env = gym.make(minerl_env_str + "-v0")

    # Setup MineRL agent
    in_model = f"data/VPT-models/foundation-model-{model_width}x.model"
    in_weights = f"data/VPT-models/foundation-model-{model_width}x.weights"
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    minerl_agent = MineRLAgent(
        env,
        device="cpu",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    minerl_agent.load_weights(in_weights)

    # Get ImpalaCNN state_dict and save it
    state_dict = minerl_agent.policy.net.img_process.cnn.state_dict()
    th.save(state_dict, f"data/VPT-models/ImpalaCNN-{model_width}x.weights")


def load_impala_cnn_weights(
    model_width=1,
    weights_path="data/VPT-models/ImpalaCNN-1x.weights",
):
    """Load previously saved"""

    if model_width == 1:
        chans=(64, 128, 128)
    elif model_width == 2:
        chans=(128, 256, 256)
    elif model_width == 3:
        chans=(192, 384, 384)
    else:
        raise ValueError(f"There is no VPT model with width {model_width}!")

    # Load state dict
    state_dict = th.load(f"data/VPT-models/ImpalaCNN-{model_width}x.weights")

    # Create model object
    impala_cnn = ImpalaCNN(
        inshape=[128, 128, 3],
        chans=chans,
        outsize=256,
        nblock=2,
        first_conv_norm=False,
        post_pool_groups=1,
        dense_init_norm_kwargs={"batch_norm": False, "layer_norm": True},
        init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
    )

    # Load state dict into model
    impala_cnn.load_state_dict(state_dict)

    return impala_cnn
