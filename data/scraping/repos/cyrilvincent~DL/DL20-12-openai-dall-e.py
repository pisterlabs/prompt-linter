from openai import OpenAI

with open("data/chatgpt/openai.env") as f:
    key = f.read()

client = OpenAI(api_key=key)

response = client.images.generate(
  model="dall-e-3",
  prompt="Un python autour d'un ordinateur des années 80",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
print(image_url)
