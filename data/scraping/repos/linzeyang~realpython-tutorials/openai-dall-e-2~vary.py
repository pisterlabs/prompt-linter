""" variant """

import json
import os
import sys
from base64 import b64decode
from pathlib import Path

import click
import openai
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent / "responses"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@click.command()
@click.option("--filename", prompt="Enter file name", help="Input file name")
@click.option("--number", prompt="Enter number", help="Number of images")
def main(filename: str, number: str) -> None:
    """ main func """

    filepath = DATA_DIR / filename

    while not filepath.exists() or not filepath.is_file():
        filename = input("Not a file or path doesn't exist! Enter filename: ")
        filepath = DATA_DIR / filename

    while not number.isdigit():
        number = input("Not a number! Enter number: ")

    with filepath.open("r", encoding="utf-8") as json_file:
        saved_response = json.load(json_file)

    image_data = b64decode(saved_response["data"][0]["b64_json"])

    try:
        response = openai.Image.create_variation(
            image=image_data, n=int(number), size="256x256", response_format="b64_json"
        )
    except openai.OpenAIError as excp:
        print(f"HTTP status: {excp.http_status}; {excp.error}")
        sys.exit(-1)

    new_file = (
        DATA_DIR / f'vary_{filepath.stem}_{response["created"]}.json'  # type:ignore
    )

    with new_file.open("w", encoding="utf-8") as file:
        json.dump(response, file)


if __name__ == "__main__":
    main()
