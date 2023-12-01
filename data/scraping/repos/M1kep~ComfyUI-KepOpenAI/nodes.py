from typing import Tuple

import torch
from openai import Client as OpenAIClient

from .lib import credentials, image


class ImageWithPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Image": ("IMAGE", {}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Generate a high quality caption for the image. The most important aspects of the image should be described first. If needed, weights can be applied to the caption in the following format: '(word or phrase:weight)', where the weight should be a float less than 2.",
                    },
                ),
                "max_tokens": ("INT", {"min": 1, "max": 2048, "default": 77}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_completion"

    CATEGORY = "OpenAI"

    def __init__(self):
        self.open_ai_client: OpenAIClient = OpenAIClient(
            api_key=credentials.get_open_ai_api_key()
        )

    def generate_completion(
        self, Image: torch.Tensor, prompt: str, max_tokens: int
    ) -> Tuple[str]:
        b64image = image.pil2base64(image.tensor2pil(Image))
        response = self.open_ai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64image}"},
                        },
                    ],
                }
            ],
        )
        if len(response.choices) == 0:
            raise Exception("No response from OpenAI API")

        return (response.choices[0].message.content,)
