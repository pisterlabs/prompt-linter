import os
from dotenv import load_dotenv
from langchain.llms import GooglePalm
from llm.config.palm_model_config import *

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


def build_text_model(temperature: float = 0.5) -> GooglePalm:
    """."""
    return GooglePalm(
        model=text_model["model_name"],
        temperature=temperature,
        max_output_tokens=text_model["parameters"]["max_output_tokens"],
        top_k=text_model["parameters"]["top_k"],
        top_p=text_model["parameters"]["top_p"],
        google_api_key=GOOGLE_API_KEY
    )


def build_chat_model(temperature: float = 0.2) -> GooglePalm:
    """."""
    return GooglePalm(
        model=chat_model["model_name"],
        temperature=temperature,
        max_output_tokens=chat_model["parameters"]["max_output_tokens"],
        top_k=chat_model["parameters"]["top_k"],
        top_p=chat_model["parameters"]["top_p"],
        google_api_key=GOOGLE_API_KEY
    )