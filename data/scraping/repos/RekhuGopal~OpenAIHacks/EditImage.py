import os
import openai

os.environ["OPENAI_API_KEY"] = "Your Open AI API Key"
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Image.create_edit(
  image=open("sunlit_lounge.png", "rb"),
  mask=open("mask.png", "rb"),
  prompt="A sunlit indoor lounge area with a pool containing a flamingo",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']