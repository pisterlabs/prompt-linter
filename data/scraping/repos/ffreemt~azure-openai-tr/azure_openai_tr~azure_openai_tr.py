"""Translate via azure openai chatgpt."""
import os
from typing import Optional

from dotenv import dotenv_values, load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from loguru import logger

# import openai

load_dotenv()  # wont override ENV

# warn if one of 'OPENAI_BASE_URL', 'OPENAI_API_KEY', 'DEPLOYMENT_NAME' is not set
if not all(map(os.getenv, dotenv_values())):
    for key in dotenv_values():
        if not os.getenv(key):
            logger.warning(
                f"os.environ['{key}'] is not set in .env or via set {key}=..."
            )

# os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"


def azure_openai_tr(
    text: str,
    to_lang: Optional[str] = None,
    temperature: float = 0.3,
    template: Optional[str] = None,
) -> str:
    """Translate via azure openai chatgpt.

    Args:
        text: to be translated
        to_lang: target language
        temperature: more focused for smaller values
        template: must contain {to_lang} and {text}

    Returns:
        translated text
    """
    if to_lang is None:
        to_lang = "Chinese"
    if template is None:
        template = """Translate the following text to {to_lang}:
            {text}"""

    if "{to_lang}" not in template or "{text}" not in template:
        logger.warning(
            f" template provided ({template}) does not "
            "contain {{to_lang}} or {{text}}. "
            "This will result a problem later on. "
            "Fix the template and try again. Exiting..."
        )
        raise SystemExit(1)

    model = AzureChatOpenAI(
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        deployment_name=os.getenv("DEPLOYMENT_NAME"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        # openai_api_type=os.getenv("OPENAI_API_TYPE"),
        temperature=temperature,
    )

    try:
        out = model(
            [HumanMessage(content=template.format(to_lang=to_lang, text=text))]
        )
    except Exception as exc:
        logger.error(exc)
        raise

    return out.content
