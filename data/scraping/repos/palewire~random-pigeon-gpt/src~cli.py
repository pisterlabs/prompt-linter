"""Use AI to generate images of New York City pigeons using a random adjective."""
from __future__ import annotations

import io
import os
from base64 import b64decode
from pathlib import Path

import click
import openai
from mastodon import Mastodon
from PIL import Image
from rich import print
from wonderwords import RandomWord


@click.command()
@click.option("--output", "-o", default="./img/")
def cli(output):
    """Use AI to generate an image of a New York City pigeon using a random adjective."""
    # Set our output directory
    output_dir = Path(output)

    # Get a random adjective that hasn't already been done
    adjective = get_random_adjective()

    # Get an image
    prompt = f"A portrait of a {adjective} pigeon in New York City"

    # Get the image
    print(f"Generating image with the following prompt: '{prompt}'")
    image = get_pigeon_image(prompt)

    # Compose the output path
    filename = f"{adjective}.png"
    filepath = output_dir / filename

    # Make sure the parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write it out
    print(f"Saving image to {filepath}")
    image.save(filepath)

    # Post to Mastodon
    print("Posting to Mastodon")
    api = Mastodon(
        client_id=os.getenv("MASTODON_CLIENT_KEY"),
        client_secret=os.getenv("MASTODON_CLIENT_SECRET"),
        access_token=os.getenv("MASTODON_ACCESS_TOKEN"),
        api_base_url="https://mastodon.palewi.re",
    )
    media_obj = api.media_post(
        filepath,
        description=f"A fictional image created by OpenAI's DALL-E 3 text-to-image model when given the following prompt: '{prompt}'",
    )
    post = api.status_post(adjective.capitalize(), media_ids=media_obj["id"])
    print(f"Posted to Mastodon: '{post['url']}'")


def get_random_adjective() -> str:
    """Get a random adjective.

    Returns
    -------
    A string word
    """
    # Get a random adjective
    r = RandomWord()
    return r.word(include_parts_of_speech=["adjectives"])


def get_pigeon_image(prompt: str) -> Image:
    """Generate an image of a pigeon in New York City.

    Args
    ----
    prompt (str)
        The instructions to pass to to OpenAI's API

    Returns
    -------
    A PIL Image object that's ready to work with.
    """
    # Connect to OpenAI
    client = openai.OpenAI()

    # Request an image from the API
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1792x1024",
        quality="hd",
        style="natural",
        n=1,
        response_format="b64_json",
    )

    # Write to an in-memory PIL object
    data = response.data[0].b64_json
    assert isinstance(data, str)
    bytes = b64decode(data)
    image = Image.open(io.BytesIO(bytes))

    # Return that object
    return image


if __name__ == "__main__":
    cli()
