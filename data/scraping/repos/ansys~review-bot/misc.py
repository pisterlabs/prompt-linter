# Copyright (c) 2023 ANSYS, Inc. All rights reserved
"""Miscellaneous functions."""
import json
import logging
import os
import re
from typing import List, Union

from openai import AzureOpenAI, OpenAI

from review.bot.defaults import ACCESS_TOKEN, API_BASE, API_TYPE, API_VERSION
from review.bot.exceptions import ValidationErrorException
from review.bot.schema import validate_output

LOG = logging.getLogger(__name__)
LOG.setLevel("DEBUG")


def _get_gh_token():
    """Return the github access token from the GITHUB_TOKEN environment variable."""
    access_token = os.environ.get("GITHUB_TOKEN")
    if access_token is None:
        raise OSError('Missing "GITHUB_TOKEN" environment variable')
    return access_token


def get_client(config_file: str = None) -> Union[OpenAI, AzureOpenAI]:
    """Get the OpenAI client with the configuration file initialization.

    Parameters
    ----------
    config_file : str, optional
        Initialization parameters of the client, by default None.

    Returns
    -------
    Union[OpenAI, AzureOpenAI]
        Initialized OpenAI client.

    Raises
    ------
    OSError
        Thrown if access token is missing.
    """
    if config_file is not None:
        with open(config_file) as json_file:
            config = json.load(json_file)

    api_type = (
        config["OPENAI_API_TYPE"]
        if config_file is not None and "OPENAI_API_TYPE" in config
        else API_TYPE
    )
    api_version = (
        config["OPENAI_API_VERSION"]
        if config_file is not None and "OPENAI_API_VERSION" in config
        else API_VERSION
    )
    api_base = (
        config["OPENAI_API_BASE"]
        if config_file is not None and "OPENAI_API_BASE" in config
        else API_BASE
    )
    access_token = (
        config["OPEN_AI_TOKEN"]
        if config_file is not None and "OPEN_AI_TOKEN" in config
        else ACCESS_TOKEN
    )

    if access_token is None:
        raise OSError('Missing "OPEN_AI_TOKEN" environment variable')

    if api_type == "azure":
        client = AzureOpenAI(azure_endpoint=api_base, api_key=access_token, api_version=api_version)
    else:
        client = OpenAI(api_key=access_token)
    return client


def open_logger(loglevel="DEBUG", formatstr="%(name)-20s - %(levelname)-8s - %(message)s"):
    """Start logging to standard output.

    Parameters
    ----------
    loglevel : str, optional
        Standard logging level. One of the following:

        - ``"DEBUG"`` (default)
        - ``"INFO"``
        - ``"WARNING"``
        - ``"ERROR"``
        - ``"CRITICAL"``

    formatstr : str, optional
        Format string.  See :class:`logging.Formatter`.

    Returns
    -------
    logging.RootLogger
        Root logging object.

    Examples
    --------
    Output logging to stdout at the ``'INFO'`` level.

    >>> import review.bot as review_bot
    >>> review_bot.open_logger('INFO')

    """
    # don't add another handler if log has already been initialized
    if hasattr(open_logger, "log"):
        open_logger.log.handlers[0].setLevel(loglevel.upper())
    else:
        log = logging.getLogger()
        ch = logging.StreamHandler()
        ch.setLevel(loglevel.upper())

        ch.setFormatter(logging.Formatter(formatstr))
        log.addHandler(ch)
        open_logger.log = log

    return open_logger.log


def add_line_numbers(patch):
    """
    Add line numbers to the added lines in a given patch string.

    The function takes a patch string and adds line numbers to the lines that
    start with a '+'. It returns a new patch string with added line numbers.
    Line numbers are added immediately to the left of any '+'.

    Parameters
    ----------
    patch : str
        The patch string containing the changes in the file.

    Returns
    -------
    str
        The modified patch string with line numbers added to the added lines.

    Examples
    --------
    >>> patch = '''@@ -1,3 +1,5 @@
    ... +from itertools import permutations
    ... +
    ... import numpy as np
    ... import pytest'''
    >>> add_line_numbers(patch)
    '@@ -1,3 +1,5 @@
    1   +from itertools import permutations
    ... +
    ... import numpy as np
    ... import pytest'

    """
    lines = patch.splitlines()
    output_lines = []
    current_line = 0

    for line in lines:
        if line.startswith("@@ "):
            # Extract the new range (one with the +)
            new_range = int(line.split("+")[1].split(",")[0])

            # Update the current line number
            current_line = new_range
            output_lines.append(line)
        else:
            if line.startswith("+"):
                output_lines.append(f"{current_line: <5}{line}")
            else:
                output_lines.append(line)

            # Increment line number only if not starting with '-'
            if not line.startswith("-"):
                current_line += 1

    return "\n".join(output_lines)


def clean_string(input_text: str):
    """Clean ``type`` and ``lines`` strings.

    Clean `type` and `lines` strings in the LLM output, in
    case some unwanted characters are mixed with the desired
    content.

    Parameters
    ----------
    input_text : str
        Raw text from the LLM.

    Returns
    -------
    str
        Cleaned text.
    """
    output = input_text.replace("[", "")
    output = output.replace("]", "")
    output = output.replace(",", "")
    output = output.replace(" ", "")
    output = output.replace("(", "")
    output = output.replace(")", "")
    return output


def clean_content(raw_content: List, text_block=None):
    """Join the list of the content.

    Join the list of the content, that might be split
    due to preprocessing, and cleans starting commas if they appear.

    Parameters
    ----------
    raw_content : list
        List with the content of the suggestion.

    Returns
    -------
    str
        Content processed.
    """
    content_string = "".join(raw_content)
    if len(content_string) == 0:
        if text_block is not None:
            ValidationErrorException("Message content is empty", text_block)
        else:
            ValidationErrorException("Message content is empty")
    if content_string[0] == ",":
        # remove comma and space
        content_string = content_string[2:]
    return content_string


def parse_suggestions(text_block: str):
    """Parse a given text block containing suggestions.

    Returns a list of dictionaries with keys: filename, lines, type, and text.

    Parameters
    ----------
    text_block : str
        The text block containing suggestions.

    Returns
    -------
    list of dict
        A list of dictionaries containing information about each suggestion.

    Examples
    --------
    >>> tblock = '''
    ... [tests/test_geometric_objects.py], [259-260], [SUGGESTION]: Replace `Rectangle` with `Quadrilateral` for clarity and consistency with the name of the class being tested.
    ... '''
    >>> parse_suggestions(tblock)
    [{'filename': 'tests/test_geometric_objects.py', 'lines': '259-260', 'type': 'SUGGESTION', 'text': 'Replace `Rectangle` with `Quadrilateral` for clarity and consistency with the name of the class being tested.'}]
    """
    suggestions = []
    pattern = "GLOBAL|COMMENT|SUGGESTION|INFO"
    # splits each individual suggestion:
    splitted_text = text_block.split("\n[")
    for suggestion_text in splitted_text:
        suggestion_info = suggestion_text.split("]")
        if len(suggestion_info) > 3:
            if "." in suggestion_info[0]:
                filename = clean_string(suggestion_info[0])
                # match if type is in position 1
                match_type1 = re.search(pattern, suggestion_info[1])
                # match if type is in position 2
                match_type2 = re.search(pattern, suggestion_info[2])
                if match_type1:
                    lines = ""
                    suggestion_type = clean_string(suggestion_info[1])
                    content = clean_content(suggestion_info[2:], text_block)
                elif match_type2:
                    lines = clean_string(suggestion_info[1])
                    suggestion_type = clean_string(suggestion_info[2])
                    content = clean_content(suggestion_info[3:], text_block)
                else:
                    LOG.warning("Output is malformed.")
                    continue
                suggestion = {
                    "filename": filename,
                    "lines": lines,
                    "type": suggestion_type,
                    "text": content,
                }
                schema_path = os.path.join(
                    os.path.dirname(__file__), "schema", "resources", "suggestion.json"
                )
                if validate_output(suggestion, schema_path):
                    suggestions.append(suggestion)
            else:
                LOG.warning(
                    "Suggestion does not contain a path to a file in the proper position, it will be ignored."
                )
        else:
            LOG.warning("Suggestion is missing some section, this suggestion will be ignored.")
    if not validate_output(suggestions):
        raise ValidationErrorException("Output format is not well formed.", llm_output=text_block)
    if len(suggestions) == 0:
        raise ValidationErrorException(
            "Output is empty due to all suggestions being malformed.",
            llm_output=text_block,
        )
    return suggestions
