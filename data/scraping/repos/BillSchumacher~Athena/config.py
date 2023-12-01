import os

import openai

from athena.llm.openai.constants import OPENAI_CHAT_COMPLETION_MODELS

# Load the API key from the environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]

MODEL = os.getenv(
    "MODEL",
)
