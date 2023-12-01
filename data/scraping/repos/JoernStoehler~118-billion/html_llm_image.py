import os
import logging
import openai
import requests
from human import Human

def html_llm_image(human: Human, size = "1024x1024"):
    if os.path.exists(human.files.image):
        logging.info(f"Image already exists at {human.files.image}")
        return

    prompt = human.vars_html["Image Prompt"]
    kwarg = {
        "prompt": prompt,
        "size": size,
        "n": 1,
        "response_format": "url",
    }
    logging.info("openai.Image.create")
    logging.info(f"{kwarg}")
    # https://platform.openai.com/docs/api-reference/images/create
    response = openai.Image.create(**kwarg)
    logging.info(response)

    cost_estimate = {
        "1024x1024": 0.020, # $ per image
        "512x512": 0.018, # $ per image
        "256x256": 0.016, # $ per image
    }[size]
    logging.info(f"Cost Estimate: ${cost_estimate:.3f}")

    # save image
    url = response["data"][0]["url"]
    logging.info(f"Downloading image from {url} to {human.files.image}")
    r = requests.get(url)
    with open(human.files.image, "wb") as f:
        f.write(r.content)