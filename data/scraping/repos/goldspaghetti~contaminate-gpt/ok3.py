import requests
from PIL import Image
from openai import OpenAI
from config import OPEN_AI_KEY

client = OpenAI(api_key=OPEN_AI_KEY)

response = client.images.generate(
    prompt="a white siamese cat",
    size="256x256",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
image_response = requests.get(image_url)
image_data = image_response.content

with open("generated_image.jpg", "wb") as f:
    f.write(image_data)

# generated_image = Image.open("generated_image.jpg")
# generated_image.show()
