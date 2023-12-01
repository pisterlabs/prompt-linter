import openai
import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_type = "azure"
openai.api_key = os.environ['AZURE_OPENAI_API_KEY']
openai.api_base = os.environ['AZURE_OPENAI_API_BASE']
openai.api_version = '2023-06-01-preview'


def get_image(prompt):
    # Retrieve the generated image
    response = openai.Image.create(
        prompt=prompt,
        size='512x512',
        n=4
    )
    image_url = response["data"][0]["url"]  # extract image URL from response
    return image_url
