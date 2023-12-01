import os
import json
import openai
from pathlib import Path
from base64 import b64decode
from dotenv import load_dotenv
from asgiref.sync import sync_to_async

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Generates 512x512 image and Save to a file
# return the path of the image as a str
async def draw(prompt) -> str:
    data_dir = Path.cwd()
    data_dir.mkdir(exist_ok=True)

    response = await sync_to_async(openai.Image.create)(
        prompt=prompt,
        n=1,
        size="512x512",
        response_format="b64_json",
    )

    file_name = data_dir / f"{prompt[:5]}-{response['created']}.json"

    with open(file_name, mode="w", encoding="utf-8") as file:
        json.dump(response, file)

    path = await convert(file_name)

    return str(path)


async def convert(path):
    data_dir = Path.cwd() / "responses"
    json_file = data_dir / path
    image_dir = Path.cwd() / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    with open(json_file, mode="r", encoding="utf-8") as file:
        response = json.load(file)

    for index, image_dict in enumerate(response["data"]):
        image_data = b64decode(image_dict["b64_json"])
        image_file = image_dir / f"{json_file.stem}-{index}.png"

        with open(image_file, mode="wb") as png:
            png.write(image_data)
        # Delete unnecessary json files
        os.remove(path)
    return image_file
