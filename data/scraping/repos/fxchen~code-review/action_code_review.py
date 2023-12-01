#!/usr/bin/env python3
import os
import sys
import openai
import string
import re
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from typing import List

FILENAME_VALID_CHARS = "-_.() %s%s" % (string.ascii_letters, string.digits)
GIT_DIFF_FILENAME_REGEX_PATTERN = r"\+\+\+ b/(.*)"
DEFAULT_OPENAI_MODEL = "gpt-4-1106-preview"
DEFAULT_ANTHROPIC_MODEL = "claude-instant-1.2"
DEFAULT_STYLE = "concise"
DEFAULT_PERSONA = "kent_beck"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 4096

OPENAI_ERROR_NO_RESPONSE = "No response from OpenAI. wtf Error:\n"
OPENAI_ERROR_FAILED = "OpenAI failed to generate a review. Error:\n"

API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

REQUEST = "Reply on how to improve the code (below). Think step-by-step. Give code examples of specific changes\n"

STYLES = {
    "zen": "Format feedback in the style of a zen koan",
    "concise": "Format feedback concisely with numbered list",
}

PERSONAS = {
    "developer": "You are an experienced software developer in a variety of programming languages and methodologies. You create efficient, scalable, and fault-tolerant solutions",
    "kent_beck": "You are Kent Beck. You are known for software design patterns, test-driven development (TDD), and agile methodologies",
    "marc_benioff": "You are Marc Benioff, internet entrepreneur and experienced software developer",
    "yoda": "You are Yoda, legendary Jedi Master. Speak like Yoda",
}


def call_openai_api(kwargs: dict) -> str:
    """
    Call the OpenAI API using the given kwargs.

    Args:
      kwargs: dict, parameters for the API call

    Returns:
      str: The response text from the API call

    Raises:
      Exception: If the API call fails
    """
    try:
        response = openai.chat.completions.create(**kwargs)
        if response.choices:
            if "text" in response.choices[0]:
                return response.choices[0].text.strip()
            else:
                return response.choices[0].message.content.strip()
        else:
            return OPENAI_ERROR_NO_RESPONSE + response.text
    except Exception as e:
        raise Exception(
            f"OpenAI API call failed with parameters {kwargs}. Error: {e}"
        )


def call_anthropic_api(kwargs: dict) -> str:
    """
    Call the Anthropic API using the given kwargs.

    Args:
      kwargs: dict, parameters for the API call

    Returns:
      str: The response text from the API call

    Raises:
      Exception: If the API call fails
    """
    try:
        anthropic = Anthropic()
        response = anthropic.completions.create(**kwargs)
        return response.completion.strip()
    except Exception as e:
        print(f"Anthropic API call failed with parameters {kwargs}. Error: {e}")
        raise Exception(
            f"Anthropic API call failed with parameters {kwargs}. Error: {e}"
        )


def prepare_openai_kwargs(
    model: str, prompt: str, max_tokens: int, temperature: float
) -> dict:
    """
    Prepares the keyword arguments for the OpenAI API call.

    Args:
      model: str, the model to use for the API call
      prompt: str, the prompt to use for the API call
      max_tokens: int, the maximum number of tokens to use for the API call
      temperature: float, the temperature to use for the API call

    Returns:
      dict: The keyword arguments for the API call
    """
    kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": prompt}],
    }
    return kwargs


def prepare_anthropic_kwargs(
    model: str, prompt: str, max_tokens: int, temperature: float
) -> dict:
    """
    Prepares the keyword arguments for the Claude API call.

    Args:
      prompt: str, the prompt to use for the API call
      model: str, the model to use for the API call
      max_tokens: int, the maximum number of tokens to use for the API call

    Returns:
      dict: The keyword arguments for the API call
    """

    kwargs = {
        "model": model,
        "max_tokens_to_sample": max_tokens,
        "prompt": f"{HUMAN_PROMPT}\n{prompt}\n{AI_PROMPT}",
    }
    return kwargs


def validate_filename(filename: str) -> bool:
    """
    Validates a filename by checking for directory traversal and unusual characters.

    Args:
      filename: str, filename to be validated

    Returns:
      bool: True if the filename is valid, False otherwise
    """
    # Check for directory traversal
    if ".." in filename or "/" in filename:
        return False

    # Check for unusual characters
    for char in filename:
        if char not in FILENAME_VALID_CHARS:
            return False

    return True


def extract_filenames_from_diff_text(diff_text: str) -> List[str]:
    """
    Extracts filenames from git diff text using regular expressions.

    Args:
      diff_text: str, git diff text

    Returns:
      List of filenames
    """
    filenames = re.findall(GIT_DIFF_FILENAME_REGEX_PATTERN, diff_text)
    sanitized_filenames = [fn for fn in filenames if validate_filename(fn)]
    return sanitized_filenames


def format_file_contents_as_markdown(filenames: List[str]) -> str:
    """
    Iteratively goes through each filename and concatenates
    the filename and its content in a specific markdown format.

    Args:
      filenames: List of filenames

    Returns:
      Formatted string
    """
    formatted_files = ""
    for filename in filenames:
        try:
            with open(filename, "r") as file:
                file_content = file.read()
            formatted_files += f"\n{filename}\n```\n{file_content}\n```\n"
        except Exception as e:
            print(f"Could not read file {filename}: {e}")
    return formatted_files


def get_prompt(
    diff: str,
    persona: str,
    style: str,
    include_files: bool,
    filenames: List[str] = None,
) -> str:
    """
    Generates a prompt for use with an LLM

    Args:
      diff: str, the git diff text
      persona: str, the persona to use for the feedback
      style: str, the style of the feedback
      include_files: bool, whether to include file contents in the prompt
      filenames: List[str], optional list of filenames to include in the prompt

    Returns:
      str: The generated prompt
    """

    prompt = f"{persona}.{style}.{REQUEST}\n{diff}"

    # Optionally include files from the diff
    if include_files:
        if filenames is None:
            filenames = extract_filenames_from_diff_text(diff)
        if filenames:
            formatted_files = format_file_contents_as_markdown(filenames)
            prompt += formatted_files

    return prompt


def main():
    # Get environment variables
    api_to_use = os.environ.get(
        "API_TO_USE", "openai"
    )  # default to OpenAI if not specified
    persona = PERSONAS.get(os.environ.get("PERSONA", DEFAULT_PERSONA))
    style = STYLES.get(os.environ.get("STYLE", DEFAULT_STYLE))
    include_files = os.environ.get("INCLUDE_FILES", "false") == "true"

    API_FUNCTIONS = {
        "openai": (
            prepare_openai_kwargs,
            call_openai_api,
            DEFAULT_OPENAI_MODEL,
        ),
        "anthropic": (
            prepare_anthropic_kwargs,
            call_anthropic_api,
            DEFAULT_ANTHROPIC_MODEL,
        ),
    }

    # Check if the specified API is supported
    if api_to_use not in API_FUNCTIONS:
        print(api_to_use)
        raise ValueError(
            f"Invalid API: {api_to_use}. Expected one of {list(API_FUNCTIONS.keys())}."
        )

    # Make sure the necessary environment variable is set
    api_key_env_var = API_KEYS.get(api_to_use)
    if api_key_env_var is None or api_key_env_var not in os.environ:
        print(f"The {api_key_env_var} environment variable is not set.")
        sys.exit(1)

    # Set API key for openai (Anthropic does so by environment variable)
    if api_to_use == "openai":
        openai.api_key = os.environ[api_key_env_var]

    # Read in the diff
    diff = sys.stdin.read()

    # Generate the prompt
    prompt = get_prompt(diff, persona, style, include_files)

    # Get the functions to prepare kwargs and call API for the specified API
    prepare_kwargs_func, call_api_func, model = API_FUNCTIONS[api_to_use]

    # Prepare kwargs for the API call
    kwargs = prepare_kwargs_func(model, prompt, LLM_MAX_TOKENS, LLM_TEMPERATURE)

    # Call the API and print the review text
    review_text = call_api_func(kwargs)

    print(f"{review_text}")


if __name__ == "__main__":
    main()
