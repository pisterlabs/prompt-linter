import logging
import os
import time
from typing import Dict, List, Union

from anthropic import AI_PROMPT, HUMAN_PROMPT, ApiException, Client

from src.models.base_model import BaseModel

CHAT_PROMPT_TEMPLATE = {"role": "Human", "content": ""}
# TEXT_PROMPT_TEMPLATE is just a simple string
_MAX_RETRIES = 3
_RETRY_TIMEOUT = 3


class AnthropicTextModels(BaseModel):
    CLAUDE_V1 = "claude-v1"


class AnthropicChatModels(BaseModel):
    CLAUDE_V1 = "claude-v1"


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.INFO,
)


def _escape_nl(prompt: str) -> str:
    """Escape new lines for error reporting."""
    return prompt.replace("\n", "\\n")


def generate_completion(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    model: Union[str, AnthropicTextModels] = AnthropicTextModels.CLAUDE_V1,
) -> str:
    """
    Send the prompt to the Anthropic completion API.

    :raises ValueError: If the prompt does not start with anthropic.HUMAN_PROMPT and end with anthropic.AI_PROMPT
    :return: The completion text.
    API docs: https://console.anthropic.com/docs/api/reference
    Examples: https://github.com/anthropics/anthropic-sdk-python/tree/4187c65ae9713b7579fcb15aec43f78ed69b97c4/examples
    """
    if isinstance(model, str):
        model = AnthropicTextModels(model)

    if not prompt.startswith(HUMAN_PROMPT):
        raise ValueError(
            f"Prompt must start with `{_escape_nl(HUMAN_PROMPT)}` but was `{_escape_nl(prompt)}`"
        )
    if not prompt.endswith(AI_PROMPT):
        raise ValueError(
            f"Prompt must end with `{_escape_nl(AI_PROMPT)}` but was `{_escape_nl(prompt)}`"
        )

    api_key = os.environ["ANTHROPIC_API_KEY"]
    for _ in range(_MAX_RETRIES):
        try:
            client = Client(api_key)
            response = client.completion(
                model=model.value,
                prompt=prompt,
                stop_sequences=[HUMAN_PROMPT],
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
            )
            return response["completion"]
        except ApiException:
            logger.warning("API Error. Sleep and try again.")
            time.sleep(_RETRY_TIMEOUT)
        except KeyError as e:
            logger.exception(e)
            logger.warning("Unexpected response format. Sleep and try again.")
            time.sleep(_RETRY_TIMEOUT)

    logger.error("Reached retry limit and did not obtain proper response")
    return model.invalid_response


def _convert_gpt_roles(
    prompt_turns: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Convert "role": user to "role": Human and "role": assistant to "role": Assistant
    """
    new_turns = []
    for turn in prompt_turns:
        if turn["role"] in ["user", "system"]:
            new_turns.append({"role": "Human", "content": turn["content"]})
        elif turn["role"] == "assistant":
            new_turns.append({"role": "Assistant", "content": turn["content"]})

    return new_turns


def format_chat_prompt(
    prompt_turns: List[Dict[str, str]],
) -> str:
    """
    Convert a list of turns into the format expected by Anthropic's default chat model ("claude-v1").
    The expected format looks like this:
    `\n\nHuman: How much is 3 times 4?\n\nAssistant:"

    We expect the prompt_turns in the following format:
    [
        {"role": "Human", "content": "X"},
        {"role": "Assistant", "content": "Y"},
        {"role": "Human", "content": "Z"},
    ]

    We will take the prompt above and convert it into the following text format:
    `\n\nHuman: X\n\nAssistant: Y\n\nHuman: Z\n\nAssistant:`
    I.e. we will prefix double newline, join the turns in the expected way, and append "\n\nAssistant:" to the end.

    :raises ValueError: if the prompt_turns are not in the expected format
    :return: the formatted prompt
    """

    prompt_turns = _convert_gpt_roles(prompt_turns)
    if any(
        (role := turn["role"]) not in ["Human", "Assistant"] for turn in prompt_turns
    ):
        raise ValueError(
            f"Invalid role {role} in prompt_turns"  # noqa: F821  # flakes8 thinks `role` is undefined
        )

    chat_prompt = (
        "\n\n"
        + "\n\n".join(f"{turn['role']}: {turn['content']}" for turn in prompt_turns)
        + "\n\nAssistant:"
    )
    return chat_prompt


def generate_chat_completion(
    prompt_turns: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 256,
    model: Union[str, AnthropicChatModels] = AnthropicChatModels.CLAUDE_V1,
) -> str:
    """
    Format the prompt in Anthropic's human-assistant format, and send it to the Anthropic completion API.
    API docs: https://console.anthropic.com/docs/api/reference
    Examples: https://github.com/anthropics/anthropic-sdk-python/tree/4187c65ae9713b7579fcb15aec43f78ed69b97c4/examples
    """
    if isinstance(model, str):
        model = AnthropicChatModels(model)

    prompt = format_chat_prompt(prompt_turns)
    return generate_completion(
        prompt=prompt, temperature=temperature, max_tokens=max_tokens, model=model.value
    )
