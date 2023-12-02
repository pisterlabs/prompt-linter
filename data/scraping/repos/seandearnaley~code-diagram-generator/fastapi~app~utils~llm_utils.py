""" Utility functions for the Large Language Models. """

import math

# Import Collection from typing
from typing import Any, Dict, List, Union

import tiktoken
from anthropic import Anthropic

from ..config import ANTHROPIC_AI_VENDOR, OPEN_AI_VENDOR

ValueType = Union[str, List[str], Any]
FunctionParameterProperty = Dict[str, ValueType]
FunctionParameters = Dict[str, FunctionParameterProperty]
FunctionType = Dict[str, Union[str, FunctionParameters]]


def anthropic_sync_count_tokens(text: str) -> int:
    """Count the number of tokens in a text string using the Anthropic API."""
    client = Anthropic()
    number_of_tokens = client.count_tokens(text)
    return number_of_tokens


def num_tokens_from_string(text: str, llm_vendor: str = OPEN_AI_VENDOR) -> int:
    """
    Returns the number of tokens in a text string.
    NOTE: openAI and Anthropics have different token counting mechanisms.
    https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
    """
    is_anthropic = llm_vendor == ANTHROPIC_AI_VENDOR

    num_tokens = (
        anthropic_sync_count_tokens(text)
        if is_anthropic
        else len(tiktoken.get_encoding("gpt2").encode(text))
    )
    return num_tokens


def estimate_word_count(num_tokens: int) -> int:
    """
    Given the number of GPT-2 tokens, estimates the real word count.
    """
    # The average number of real words per token for GPT-2 is 0.56, according to OpenAI.
    # Multiply the number of tokens by this average to estimate the total number of real
    # words.
    return math.ceil(num_tokens * 0.56)


def validate_max_tokens(max_tokens: int) -> None:
    """
    Validate the max_tokens argument, raising a ValueError if it is not valid.
    """
    if max_tokens <= 0:
        raise ValueError("The input max_tokens must be a positive integer.")


def calculate_property_tokens(
    properties: FunctionParameterProperty, encoding: Any
) -> int:
    """Calculate tokens for property fields."""
    tokens = 0
    for field, value in properties.items():
        if field in ("type", "description"):
            tokens += 2 + len(encoding.encode(value))
        elif field == "enum":
            tokens += sum(3 + len(encoding.encode(o_e)) for o_e in value) - 3
    return tokens


def calculate_function_tokens(function: FunctionType, encoding: Any) -> int:
    """Calculate tokens for function."""
    tokens = len(encoding.encode(function["name"])) + len(
        encoding.encode(function["description"])
    )
    parameters = function.get("parameters", {})
    properties = (
        parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    )

    for key, values in properties.items():
        if isinstance(values, dict):
            tokens += len(encoding.encode(key)) + calculate_property_tokens(
                values, encoding
            )

    tokens += 11
    return tokens


def num_tokens_from_functions(
    functions: List[FunctionType], model: str = "gpt-3.5-turbo"
) -> int:
    """Return the number of tokens used by a list of functions."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = (
        sum(calculate_function_tokens(function, encoding) for function in functions)
        + 12
    )
    return num_tokens
