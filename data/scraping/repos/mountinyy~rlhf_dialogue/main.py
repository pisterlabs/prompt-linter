import random

import numpy as np
import torch
from omegaconf import OmegaConf

# from train.actor_train import train_actor
from train.critic_train import train_reward

# from train.actor import train_actor
from train.datasets import AnthropicDataset

"""
def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--model_name",
        required=True,
        default="facebook/bart-base",
        choices=["facebook/bart-base"],
        help="pretrained model name form huggingface",
    )

    return arg_parser.parse_args()
"""


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # args = get_args()
    conf = OmegaConf.load("./config.yaml")
    set_seed(conf.common.seed)

    # Download Dataset
    download_dataset = AnthropicDataset(conf)
    download_dataset.save_dataset()

    # SFT
    # train_actor(conf)
    # RM
    train_reward(conf)
