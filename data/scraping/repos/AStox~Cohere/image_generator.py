from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


def generate_recipe_image(recipe):
    client = OpenAI()

    response = client.images.generate(
        model="dall-e-3",
        prompt=f"""Generate a photorealistic image a meal based on the following recipe, minimalistically displayed on a plain, polished concrete blue-ish countertop:
                {recipe}""",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url
