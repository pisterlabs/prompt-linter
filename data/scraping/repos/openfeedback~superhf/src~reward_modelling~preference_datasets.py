"""
This file creates a dataset of human preference data.

Datasets are torch.utils.data.Dataset objects that can be indexed to return
a datum as a Tuple[str, str] in the form (winner_response, loser_response).

"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class AnthropicHelpfulHarmless(Dataset):
    """
    Human preference data about helpfulness and harmlessness from Anthropic.
        https://huggingface.co/datasets/Anthropic/hh-rlhf
    """

    def __init__(self, split="train", data_dir=None):
        data = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir)[split]
        self.winner_responses = data["chosen"]
        self.loser_responses = data["rejected"]

    def __getitem__(self, idx):
        return self.winner_responses[idx], self.loser_responses[idx]

    def __len__(self):
        return len(self.winner_responses)


