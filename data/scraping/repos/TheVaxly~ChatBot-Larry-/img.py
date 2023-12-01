import openai
import os

OPENAI_API_KEY = os.getenv("img")

openai.api_key = OPENAI_API_KEY

def gen(prompt):
    response = openai.Image.create(
        prompt=prompt,
        model="image-alpha-001"
    )
    image_url = response['data'][0]['url']
    return image_url