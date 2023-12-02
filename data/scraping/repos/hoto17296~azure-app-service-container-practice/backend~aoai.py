"""Azure Open AI の設定"""
from os import getenv
from logging import getLogger

logger = getLogger("uvicorn")

import openai

# Required
AZURE_OPENAI_RESOURCE = getenv("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_MODEL = getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_KEY = getenv("AZURE_OPENAI_KEY")

# Optional
AZURE_OPENAI_API_VERSION = getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
AZURE_OPENAI_TEMPERATURE = float(getenv("AZURE_OPENAI_TEMPERATURE", 0))
AZURE_OPENAI_TOP_P = float(getenv("AZURE_OPENAI_TOP_P", 1.0))
AZURE_OPENAI_MAX_TOKENS = int(getenv("AZURE_OPENAI_MAX_TOKENS", 1000))
AZURE_OPENAI_STOP_SEQUENCE = getenv("AZURE_OPENAI_STOP_SEQUENCE")

openai.api_type = "azure"
openai.api_base = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"
openai.api_version = AZURE_OPENAI_API_VERSION
openai.api_key = AZURE_OPENAI_KEY


def chat(messages):
    return openai.ChatCompletion.acreate(
        engine=AZURE_OPENAI_MODEL,
        messages=messages,
        temperature=AZURE_OPENAI_TEMPERATURE,
        max_tokens=AZURE_OPENAI_MAX_TOKENS,
        top_p=AZURE_OPENAI_TOP_P,
        stop=AZURE_OPENAI_STOP_SEQUENCE.split("|") if AZURE_OPENAI_STOP_SEQUENCE else None,
        stream=True,
    )
