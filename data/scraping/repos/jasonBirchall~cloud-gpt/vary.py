# vary.py

import json
import os
from base64 import b64encode
from pathlib import Path

import openai

import _helper_

DATA_DIR = Path.cwd() / "responses"
SOURCE_FILE = DATA_DIR / "An ec-166799484.json"

openai.api_key = os.getenv("OPENAI_API_KEY")

_helper_.check_key("OPENAI_API_KEY")

with open(SOURCE_FILE, mode="r", encoding="utf-8") as f:
    saved_response = json.load(f)
    image_data = b64encode(saved_response["data"][0]["b64_json"])

response = openai.Image.create_variation(
    image=image_data,
    n=3,
    size="256x256",
    response_format="b64_json",
)

new_file_name = f"vary-{SOURCE_FILE.stem[:5]}-{response['created']}.json"

with open(DATA_DIR / new_file_name, mode="w", encoding="utf-8") as f:
    json.dump(response, f)
