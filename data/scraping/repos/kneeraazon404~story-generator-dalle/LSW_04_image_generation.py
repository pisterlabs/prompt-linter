"""
This module provides functionality to generate images from text prompts using OpenAI's API
and post the results to a webhook.
"""

import time
import logging
import requests
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize the OpenAI client
client = OpenAI()


def post_to_webhook(response):
    """
    Posts a given response to a predefined webhook.

    :param response: The response to post, either as a string or a response object.
    """
    text_content = response if isinstance(response, str) else response.text
    payload = {"response": text_content}
    requests.post("https://webhook.site/LSW-process-logging", json=payload)


def generate_images_from_prompts(image_prompts):
    """
    Generates images for a given set of prompts and posts them to a webhook.

    :param image_prompts: A dictionary of prompts where the key is a unique identifier for the prompt.
    """
    images = {}

    for prompt_key, prompt_text in image_prompts.items():
        logging.info(f"{prompt_key} is in progress...")
        attempt = 0
        max_attempts = 3

        while attempt < max_attempts:
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt_text,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
                images[prompt_key] = response.data[0].url
                logging.info(
                    f"Generated image URL for {prompt_key}: {images[prompt_key]}"
                )
                post_to_webhook(images[prompt_key])
                break
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed for {prompt_key}: {e}")
                attempt += 1
                time.sleep(1)

            if attempt == max_attempts:
                logging.error(
                    f"Failed to generate image for {prompt_key} after {max_attempts} attempts."
                )

    logging.info("All generated image URLs:")
    for prompt_key, image_url in images.items():
        logging.info(f"{prompt_key}: {image_url}")
