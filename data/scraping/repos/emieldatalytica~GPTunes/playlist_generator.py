"""Generate a playlist with OpenAI's GPT API based on a given theme."""

import base64
import io
import os
from typing import Any

import fastapi
import openai
from fastapi import HTTPException
from openai.error import RateLimitError
from PIL import Image

from prompt import DALLE_PROMPT, INITIAL_PROMPT
from utils import Playlist, extract_json_from_response, get_logger, spotify_client

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise KeyError("OPENAI_API_KEY environment variable not set.")

playlist_generator = fastapi.FastAPI()
logger = get_logger(__name__)


@playlist_generator.get("/query_gpt_model")
def query_model_for_playlist(theme: str) -> str:
    """Generate a playlist with OpenAI's GPT API based on a given theme.

    Args:
        theme (str): The theme of the playlist.

    Returns:
        str: The model response in json-format.
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "user", "content": INITIAL_PROMPT.replace("{theme}", theme)},
        ],
        temperature=0.8,
    )
    return completion.choices[0].message["content"]  # type: ignore


@playlist_generator.get("/query_dalle_model")
def create_cover_image(playlist_title: str) -> str:
    """Generate a cover image for the playlist with OpenAI's DALL-E API.

    Args:
        playlist_title (str): The title of the playlist.

    Returns:
        str: The base64 encoded image.
    """
    logger.info(f"Creating cover image for playlist titled '{playlist_title}'.")
    image = openai.Image.create(
        prompt=DALLE_PROMPT.replace("{playlist_title}", playlist_title),
        n=1,
        size="256x256",
        response_format="b64_json",
    )

    # Decode the base64 image data and open the image
    bytes_obj = base64.b64decode(image.data[0].b64_json)
    image = Image.open(io.BytesIO(bytes_obj))

    # Resize the image and decrease quality to fit the maximum payload size that Spotify allows (256kb)
    image = image.resize((800, 800))
    image_data = io.BytesIO()
    image.save(image_data, format="JPEG", quality=75)
    image_b64 = base64.b64encode(image_data.getvalue()).decode("utf-8")

    return image_b64


def map_tracks_to_spotify_ids(tracks: dict) -> dict:
    """Map the tracks to their spotify ids using the spotify api.

    Args:
        tracks (dict): The tracks to map.

    Returns:
        dict: The tracks mapped to their spotify ids.
    """
    mapped_tracks = {}
    for artist, track in tracks.items():
        result = spotify_client.spotify.search(f"{artist} - {track}", type="track", limit=1)
        if result is None:
            # search again on only track name
            result = spotify_client.spotify.search(f"{track}", type="track", limit=1)
        if result is None:
            continue
        items = result.get("tracks", {}).get("items", [])
        if not items:
            continue
        mapped_tracks[track] = items[0]["id"]
    logger.info(f"Mapped {len(mapped_tracks.values())} tracks to spotify ids.")
    return mapped_tracks


@playlist_generator.get("/playlist", response_model=Playlist)
def compose_playlist(theme: str) -> Any:
    """Compose a playlist for a given theme.

    Args:
        theme (str): The theme of the playlist.
    Returns:
        Playlist: The composed playlist with tracks, title and description.
    """
    logger.info(f"Given theme: '{theme}'")
    try:
        gpt_response = query_model_for_playlist(theme)
    except RateLimitError:
        logger.error("OpenAI API usage limit has been hit.")
        raise HTTPException(
            status_code=429, detail="OpenAI API usage limit has been hit. Maybe ask the creator to up the limit?"
        )
    except Exception as e:
        logger.error(f"An error occurred while querying the GPT model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    playlist_json = extract_json_from_response(gpt_response)

    if not isinstance(playlist_json, dict):
        logger.error(f"The GPT response could not be processed into a structured dictionary: {playlist_json}")
        raise HTTPException(
            status_code=500,
            detail="The GPT response could not be processed into a structured dictionary. "
            "Maybe try again with a different theme?",
        )
    if not all(key in playlist_json.keys() for key in ["tracks", "title", "description"]):
        logger.error(f"The GPT response does not have keys 'tracks', 'title' and 'description': {playlist_json}")
        raise HTTPException(
            status_code=500,
            detail="The GPT response does not have keys 'tracks', 'title' and 'description'. "
            "Maybe try again with a different theme?",
        )
    logger.info(f"Extracted playlist with {len(playlist_json['tracks'].keys())} tracks from GPT response.")

    try:
        playlist_json["cover_image"] = create_cover_image(playlist_json["title"])
    except Exception as e:
        # Only log error, don't raise exception since playlist can be created without cover image
        logger.error(f"An error occurred while generating the cover image: {e}")
        playlist_json["cover_image"] = None

    track_ids = map_tracks_to_spotify_ids(playlist_json["tracks"])
    playlist_json["tracks"] = list(track_ids.values())

    playlist = Playlist(**playlist_json)
    return playlist
