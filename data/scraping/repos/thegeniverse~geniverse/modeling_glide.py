import os
import gc
import logging
from typing import *

import torch
import torchvision
import numpy as np
import PIL
from PIL import Image
from torch.utils.checkpoint import checkpoint
from geniverse_hub import hub_utils
from geniverse.modeling_utils import ImageGenerator


class Glide(
        torch.nn.Module, ):
    """
    Image generator that leverages the decoder of VQGAN to 
    optimize images. This code uses the original implementation
    from https://github.com/CompVis/taming-transformers.
    """
    def __init__(
        self,
        device: str = 'cuda',
        timestep_respacing: int = 200,
        **kwargs,
    ) -> None:
        """
        Glide model from OpenAI
        """
        super(Glide, self).__init__()

        self.device = device

        glide_hub = hub_utils.load_from_hub("glide")

        self.glide_model = glide_hub.load_model(
            device=self.device,
            timestep_respacing=100,
        )

    def generate_from_prompt(
        self,
        prompt,
        mode: str = "clip_guided",
        num_results: int = 1,
        img=None,
        img_mask=None,
        *args,
        **kwargs,
    ) -> Tuple[List[PIL.Image.Image], List[torch.Tensor]]:

        if mode == "inpaint":
            gen_img_list = self.glide_model.inpaint(
                prompt=prompt,
                img=img,
                img_mask=img_mask,
                batch_size=num_results,
            )

        elif mode == "text2im":
            gen_img_list = self.glide_model.text2im(
                prompt=prompt,
                batch_size=num_results,
            )

        elif mode == "clip_guided":
            gen_img_list = self.glide_model.clip_guided(
                prompt=prompt,
                batch_size=num_results,
            )

        return gen_img_list


if __name__ == "__main__":
    glide = Glide()
    mode = "clip_guided"
    num_results = 2

    upsample_temp = 0.997,
    guidance_scale = 10.

    # img = torchvision.transforms.PILToTensor()(
    #     Image.open("./img.jpg"))[None, :]
    # img_mask = torchvision.transforms.PILToTensor()(
    #     Image.open("./img_mask.jpg"))[None, :]

    gen_img_list = glide.generate_from_prompt(
        "Master of puppets",
        num_results=num_results,
        guidance_scale=guidance_scale,
    )

    for idx, gen_img in enumerate(gen_img_list):
        gen_img.save(f"{mode}_{idx}.png")
