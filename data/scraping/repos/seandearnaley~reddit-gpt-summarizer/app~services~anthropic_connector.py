"""Anthropic Connector"""

from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
from config import ConfigLoader
from data_types.summary import GenerateSettings
from env import EnvVarsLoader
from log_tools import Logger

config = ConfigLoader.get_config()
app_logger = Logger.get_app_logger()
env_vars = EnvVarsLoader.load_env()


@Logger.log
def complete_anthropic_text(
    prompt: str,
    max_tokens: int,
    settings: GenerateSettings,
) -> str:
    """
    Use Anthropic's GPT model to complete text based on the given prompt.

    Args:
        prompt (str): The prompt to use as the starting point for text completion.
        max_tokens (int, optional): The maximum number of tokens to generate in the
        response.
        settings (GenerateSettings): The settings to use for generating the text.

    Returns:
        str: The completed text.
    """

    try:
        anthropic_client = Anthropic(api_key=env_vars["ANTHROPIC_API_KEY"])
        response = anthropic_client.completions.create(
            prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
            stop_sequences=[HUMAN_PROMPT],
            model=settings["selected_model"],
            max_tokens_to_sample=max_tokens,
        )

        return response.completion.strip()
    except Exception as err:  # pylint: disable=broad-except
        return f"error: {err}"
