import os
import openai
from dotenv import load_dotenv
import requests
import shutil

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY") # Must have .env containing this variable and your API key in the same directory.

prompt = "a hyper realistic robot on one of saturn's moons" # Type your prompt here.

response = openai.Image.create(
  prompt=prompt,
  n=1,
  size="1024x1024"
)

image_url = response['data'][0]['url']

image = requests.get(image_url, stream=True)
desktop = os.path.expanduser("~/Desktop")

with open(os.path.join(desktop, 'image.jpg'), 'wb') as out_file: # Save's file to your desktop
    shutil.copyfileobj(image.raw, out_file)
del image
