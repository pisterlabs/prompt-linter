import os
os.environ["OPENAI_API_KEY"] = "<your-openai-api-key>"
import openai
import io
import requests
import os

def generate_dalle_images(prompt):
    os.makedirs("output", exist_ok=True)
    response = openai.Image.create(
        prompt=prompt,
        n=4,
        size="512x512"
    )
    for i in range(len(response["data"])):
        image_url = response['data'][i]['url']
        with io.BytesIO(requests.get(image_url).content) as buf:
            with open(f"output/dalle_out_{i}.png", "wb") as fp:
                fp.write(buf.getvalue())

if __name__ == "__main__":
    generate_dalle_images("a corgi playing a flame throwing trumpet")