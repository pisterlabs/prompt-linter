import openai
import os
import requests
import json
import base64
from PIL import Image

openia_key = "sk-" + os.getenv("OPENAI_KEY")
openai.api_key = openia_key

with open("champion.txt", "r") as file:
    contenu = file.read()

response = openai.Image.create(
  prompt=contenu,
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
image_response = requests.get(image_url)
image_data = image_response.content
with open("dalle_image.jpg", "wb") as image_file:
    image_file.write(image_data)
print(image_url)