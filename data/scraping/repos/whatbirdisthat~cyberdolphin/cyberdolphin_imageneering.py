from PIL import ImageOps
import torch
import numpy as np
from .openai_client import OpenAiClient, DALL_E_SIZE


class CyberDolphinImageneering:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ('STRING', {'default': 'darth vader with yoda ears', 'multiline': True}),
            "size": (["256x256", "512x512", "1024x1024"], {'default': "1024x1024"}),
        }}

    CATEGORY = "🐬 CyberDolphin"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, prompt: str, size: DALL_E_SIZE):
        i = OpenAiClient.image_create(prompt=prompt, size=size)
        # copy/pasted from class LoadImage:
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return image, mask.unsqueeze(0)
