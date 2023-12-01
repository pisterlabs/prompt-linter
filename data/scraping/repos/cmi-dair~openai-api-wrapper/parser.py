"""This module provides an ArgumentParser object for the CLI."""

import argparse
import logging
import pathlib

from openai_api_wrapper import constants, logs

LOGGER_NAME = logs.LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def get_parser() -> argparse.ArgumentParser:
    """Returns an ArgumentParser object for the CLI.

    Returns:
        argparse.ArgumentParser: An ArgumentParser object for the CLI.
    """
    parser = argparse.ArgumentParser(
        prog="OpenAI API Wrapper",
        description="OpenAI API Wrapper",
        epilog="Issues can be reported to: https://github.com/cmi-dair/openai-api-wrapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mandatory_group = parser.add_argument_group(
        "Mandatory arguments", "Mandatory arguments for all models."
    )
    chat_completion_group = parser.add_argument_group(
        "Chat Completion arguments", "Arguments for chat completion models."
    )
    optional_group = parser.add_argument_group(
        "Optional arguments", "Optional arguments for all models."
    )

    mandatory_group.add_argument(
        "model",
        type=str,
        help="The model to use for the API call",
        choices=constants.SUPPORTED_MODELS,
    )
    chat_completion_group.add_argument(
        "--message",
        type=str,
        help="A message to add to the conversation. Can be used multiple times. The first message must start with 'system'. Each subsequent message must start with 'user:' or 'assistant:'",
        action="append",
    )
    chat_completion_group.add_argument(
        "--messages-file",
        type=pathlib.Path,
        help="A file containing messages to add to the conversation. Each message must start with 'user:' or 'assistant:'",
    )
    optional_group.add_argument(
        "--api-key",
        type=str,
        help="Your OpenAI API key. If not provided, the OPENAI_API_KEY environment variable will be used.",
        default=None,
    )

    return parser
