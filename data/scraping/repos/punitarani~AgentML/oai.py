"""agentml/oai.py"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from config import PROJECT_PATH

env_loaded = load_dotenv(PROJECT_PATH.joinpath(".env"))
if not env_loaded:
    raise RuntimeError("Failed to load environment variables")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY is not None, "OPENAI_API_KEY environment variable not set"

client = OpenAI(api_key=OPENAI_API_KEY)
