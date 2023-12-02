import os
import random

import openai
import requests
from PIL import Image, ImageDraw

# Initialize OpenAI with your API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_gradient(width=500, height=500):
    """Generate a gradient image"""
    base = Image.new('RGB', (width, height), color=(0, 0, 0))
    top_color = (random.randint(0, 255), random.randint(
        0, 255), random.randint(0, 255))
    bottom_color = (random.randint(0, 255), random.randint(
        0, 255), random.randint(0, 255))
    draw = ImageDraw.Draw(base)
    for y in range(height):
        r = (top_color[0] * (height - y) + bottom_color[0] * y) / height
        g = (top_color[1] * (height - y) + bottom_color[1] * y) / height
        b = (top_color[2] * (height - y) + bottom_color[2] * y) / height
        draw.line([(0, y), (width, y)], fill=(int(r), int(g), int(b)))
    return base


def generate_ai_background(width, height, quote):
    """
    Generate a background image using DALL·E based on a given size.

    Args:
    - width (int): The width of the image.
    - height (int): The height of the image.

    Returns:
    - PIL.Image: The generated image.
    """
    # Using the maximum available size for best resolution
    size = "1024x1024"
    # add quote to prompt
    prompt = ("A background pattern, either an abstract or nature theme. The image must not have any light colors and have a darker color scheme. The theme of the image is: " + quote)

    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=size
    )

    # Extract URL from the response
    image_url = response['data'][0]['url']

    # Fetch the image from the URL and convert to PIL Image
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Resize the image to desired dimensions
    image_resized = image.resize((width, height), Image.LANCZOS)

    return image_resized
