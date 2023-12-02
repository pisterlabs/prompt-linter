from pathlib import Path

import torch
from torch import nn

from .helpers import check_download_url


# https://github.com/openai/DALL-E/blob/master/notebooks/usage.ipynb
model_url = "https://cdn.openai.com/dall-e/encoder.pkl"


class DVAE(nn.Module):
    """
    Discrete VAE from OpenAI DALL-E

        path (str): locally saved model checkpoint
        device (str or device, optional): which device to load the model checkpoint onto
    """
    def __init__(self, path='./models', device=None):
        super().__init__()

        path = Path(path)  # Make sure this is a Path object

        self.name = f"DVAE"
        self.input_channels = 3
        self.input_width = 256
        self.input_height = 256
        self.dtype = torch.int16

        self.num_features = 1024  # 32 x 32

        self.device = device

        file_path = check_download_url(path, model_url)
        self.encoder = torch.load(file_path, map_location=device)
        for param in self.encoder.parameters():  # Freeze encoder parameters
            param.requires_grad = False

    def normalization(self, x: torch.Tensor) -> torch.Tensor:
        # map_pixels
        logit_laplace_eps: float = 0.1

        if len(x.shape) != 4:
            raise ValueError('expected input to be 4d')
        if x.dtype != torch.float:
            raise ValueError('expected input to have type float')

        # 0.8 * x + 0.1
        return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps

    """
    Compute dvae image embeddings

        input (tensor [B, C, W, H]): batch of image tensors

    Returns a tensor of integer feature embeddings [B, 1024]
    where embeddings refer to a codebook
    """
    def forward(self, input):
        # Make sure input matches expected dimensions
        B, C, W, H = input.shape  # Batch size, channels, width, height
        assert (W == self.input_width) and (H == self.input_height)

        input = self.normalization(input)
        logits = self.encoder(input)  # [B x 8192 x 32 x 32]
        codes = torch.argmax(logits, dim=1)  # [B x 32 x 32]
        return codes.reshape(B, -1).to(self.dtype)  # [B x 1024]
