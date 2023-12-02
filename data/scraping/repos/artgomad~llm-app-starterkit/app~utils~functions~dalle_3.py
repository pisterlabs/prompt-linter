from dotenv import load_dotenv
import openai
from PIL import Image
import requests

load_dotenv()  # Load .env file
client = openai.OpenAI()


def generate_image(prompt, n: int = 1, size: str = "1024x1024"):

    print("Starting image generation")

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality="standard",
        n=1
    )
    print(response)

    image_url = response.data[0].url

    im = Image.open(requests.get(image_url, stream=True).raw)
    im.save("temp.png")
    print(image_url)

    return image_url, [image_url]
