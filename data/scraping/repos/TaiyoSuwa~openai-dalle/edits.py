import openai
import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

openai.api_key =  os.environ.get("API_KEY")

response = openai.Image.create_edit(
  image=open("./sample_image/original.png", "rb"),
  mask=open("./sample_image/mask.png", "rb"),
  prompt="A white tiger on the red carpet",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
print(image_url)