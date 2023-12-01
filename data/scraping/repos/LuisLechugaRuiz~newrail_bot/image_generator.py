import requests
import io
import os.path
from PIL import Image
import uuid
import openai
from base64 import b64decode

from newrail.capabilities.capability import Capability
from newrail.capabilities.utils.decorators import action_decorator
from newrail.config.config import Config


class ImageGenerator(Capability):
    """Generates images from text."""

    @action_decorator
    def generate_image(self, prompt: str):
        """Generates an image from a prompt.

        Args:
            prompt (str): The prompt to generate an image from.
        """

        working_directory = Config().permanent_storage

        filename = str(uuid.uuid4()) + ".jpg"

        # DALL-E
        if Config().image_provider == "dalle":
            openai.api_key = Config().openai_api_key

            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="256x256",
                response_format="b64_json",
            )

            print("Image Generated for prompt:" + prompt)

            image_data = b64decode(response["data"][0]["b64_json"])

            with open(working_directory + "/" + filename, mode="wb") as png:
                png.write(image_data)

            return "Saved to disk:" + filename

        # STABLE DIFFUSION # TODO: Fix this adding huggingface_api_token as env_var.
        elif Config().image_provider == "sd":
            API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
            headers = {"Authorization": "Bearer " + Config().huggingface_api_token}

            response = requests.post(
                API_URL,
                headers=headers,
                json={
                    "inputs": prompt,
                },
            )

            image = Image.open(io.BytesIO(response.content))
            print("Image Generated for prompt:" + prompt)

            image.save(os.path.join(working_directory, filename))

            return "Saved to disk:" + filename

        else:
            return "No Image Provider Set"
