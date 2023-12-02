import openai
import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

response = openai.Image.create(
  prompt="a white siamese cat with blue eyes, sitting on a wooden floor",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
print(image_url)