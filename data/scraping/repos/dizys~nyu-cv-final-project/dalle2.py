import os
import openai
from PIL import Image
import requests
from io import BytesIO

openai.api_key = os.getenv("OPENAI_API_KEY")


def make_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    elif width > height:
        left = (width - height) // 2
        right = (width + height) // 2
        return image.crop((left, 0, right, height))
    else:
        top = (height - width) // 2
        bottom = (height + width) // 2
        return image.crop((0, top, width, bottom))


def generate_variations(original_image: Image.Image, image_size: int = 512) -> Image.Image:
    original_image = make_square(original_image)
    original_image.save("/tmp/original.png")
    response = openai.Image.create_variation(
        image=open("/tmp/original.png", "rb"),
        n=1,
        size=f"{image_size}x{image_size}",
    )
    image_url = response['data'][0]['url']
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))
