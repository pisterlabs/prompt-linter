import os
import openai
from dotenv import load_dotenv

try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Failed to load .env file. Continuing without it. Error: {e}")

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in environment. Please set it in your .env file, as an environment variable, or another configuration method."
    )


def create_openai_client():
    if not openai.api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Please set it in your .env file, as an environment variable, or another configuration method."
        )
    return openai
