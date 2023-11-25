import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def imagine(prompt, n=1, size="1024x1024"):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality="standard",
        n=n,
    )
    image = requests.get(response.data[0].url)

    with open(f"images/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", "wb") as f:
        f.write(image.content)
    return (response.data[0].revised_prompt, "Image downloaded successfully!")
