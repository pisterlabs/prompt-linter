import os
import openai

os.environ["OPENAI_API_KEY"] = "Your Open AI API Key"
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Image.create_variation(
  image=open("corgi_and_cat_paw.png", "rb"),
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']