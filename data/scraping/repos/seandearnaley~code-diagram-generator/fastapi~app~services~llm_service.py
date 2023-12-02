"""Service for LLM Models"""
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import openai
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
from loguru import logger
from openai.openai_object import OpenAIObject
from pyrate_limiter import Duration, Limiter, RequestRate

from ..config import ANTHROPIC_AI_VENDOR, LLM_CONFIG_PATH
from ..exceptions import AnthropicException, LLMException, OpenAIException
from ..models import LLMConfig, LLMDefinition
from ..utils.llm_utils import validate_max_tokens

openai.organization = os.environ.get("OPENAI_ORG_ID")
openai.api_key = os.environ.get("OPENAI_API_KEY")


rate_limits = (RequestRate(60, Duration.MINUTE),)  # 60 requests a minute

# Create the rate limiter / Pyrate Limiter instance
limiter = Limiter(*rate_limits)


async def load_llm_config() -> LLMConfig:
    """Reads diagram configuration from a JSON file"""
    with LLM_CONFIG_PATH.open(encoding="utf-8") as json_file:
        data: Dict[str, Any] = json.load(json_file)
        llm_vendors = dict(data["llm_vendors"].items())

    return LLMConfig(
        llm_vendors=llm_vendors,
        llm_vendor_names=data["llm_vendor_names"],
    )


def get_llm_by_id(llm_config: LLMConfig, llm_id: str) -> LLMDefinition | None:
    """Get a llm config by id from the loaded llm configuration"""
    for _, llms in llm_config.llm_vendors.items():
        for llm in llms:
            if llm.id == llm_id:
                return llm
    return None


def complete_text(
    max_tokens: int,
    model: str,
    vendor: str,
    messages: list[dict[str, str]],
    functions: Optional[List[Any]] = None,
    callback: Optional[Callable[[Any], Union[Tuple[str, str, str], str]]] = None,
) -> Union[Tuple[str, str, str], str]:
    """LLM orchestrator"""
    logger.info(f"Starting Complete Text: messages: {messages}")
    validate_max_tokens(max_tokens)

    is_anthropic = vendor == ANTHROPIC_AI_VENDOR
    try:
        limiter.ratelimit("complete_text")

        # delegate to the appropriate completion method
        if is_anthropic:
            return complete_anthropic_text(
                max_tokens=max_tokens, model=model, messages=messages
            )

        return complete_openai_text(
            max_tokens=max_tokens,
            model=model,
            messages=messages,
            functions=functions,
            callback=callback,
        )
    except LLMException as exc:
        raise LLMException(f"Error completing text: {exc}") from exc


def complete_openai_text(
    max_tokens: int,
    model: str,
    messages: list[dict[str, str]],
    functions: Optional[List[Any]] = None,
    callback: Optional[Callable[[Any], Union[Tuple[str, str, str], str]]] = None,
) -> Union[Tuple[str, str, str], str]:
    """Use OpenAI's GPT model to complete text based on the given prompt."""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            # temperature=0.3,
            functions=functions,
            function_call="auto",
        )
        if not isinstance(response, OpenAIObject):
            raise ValueError("Invalid Response")

        if callback:
            return callback(response)

        return "Response doesn't have choices or choices have no text."

    except openai.OpenAIError as err:
        raise OpenAIException(f"OpenAI Client Error: {err}") from err
    except ValueError as err:
        raise OpenAIException(f"OpenAI Client Value error: {err}, {err.args}") from err
    except OpenAIException as err:
        raise OpenAIException(f"complete_openai_text Exception: {err}") from err


def format_anthropic_prompt(messages: list[dict[str, str]]) -> str:
    """Format the messages into a prompt for the anthropic api"""
    prompt = ""
    for message in messages:
        if message["role"] == "user":
            prompt += f"{HUMAN_PROMPT} {message['content']}"
        elif message["role"] == "assistant":
            prompt += f"{AI_PROMPT} {message['content']}"
    return prompt


def complete_anthropic_text(
    max_tokens: int,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    """Use Anthropic's model to complete text based on the given prompt."""
    try:
        anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        prompt = format_anthropic_prompt(messages)
        response = anthropic_client.completions.create(
            prompt=prompt,
            stop_sequences=[HUMAN_PROMPT],
            model=model,
            max_tokens_to_sample=max_tokens,
        )

        return response.completion.strip()
    except AnthropicException as err:
        raise AnthropicException(f"Anthropic Client Error: {err}") from err
