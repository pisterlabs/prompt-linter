import os
from typing import List

import openai
import yaml
from util import log_openai_calls, logger, openai_throttler

# Reading YAML file
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    globals().update(config)

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")


@log_openai_calls
@openai_throttler(api_call_times=3.0, rate_limit=20.0)
def create(
    engine: str,
    prompt: str,
    max_tokens: int = 2040,
    n: int = 1,
    stop: List[str] = None,
    temperature: float = 0.5,
    messages=None,
):
    if config["api_type"] == "openai":
        response = openai_create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
        )
    elif config["api_type"] == "azure":
        response = azure_openai_create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
        )
    elif config["api_type"] == "chatgpt":
        response = chatapi_create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
            messages=messages,
        )
    else:
        logger.info("Please configure api_type")

    return response


def openai_create(
    engine: str,
    prompt: str,
    max_tokens: int = 2040,
    n: int = 1,
    stop: List[str] = None,
    temperature: float = 0.5,
):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        temperature=temperature,
    )

    return response


def azure_openai_create(
    engine: str,
    prompt: str,
    max_tokens: int = 2040,
    n: int = 1,
    stop: List[str] = None,
    temperature: float = 0.5,
):
    openai.api_type = "azure"
    openai.api_base = os.environ.get("OPENAI_API_BASE")
    openai.api_version = "2022-12-01"
    logger.info("-----------------{}".format(engine))
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        temperature=temperature,
    )

    return response


def chatapi_create(
    engine: str,
    prompt: str,
    max_tokens: int = 2040,
    n: int = 1,
    stop: List[str] = None,
    temperature: float = 0.5,
    messages=None,
):
    response = openai.ChatCompletion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        temperature=temperature,
        messages=messages,
    )

    return response
