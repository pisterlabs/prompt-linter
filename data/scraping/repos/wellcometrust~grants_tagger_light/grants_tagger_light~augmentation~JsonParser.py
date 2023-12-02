"""
From langchain: https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/langchain/langchain/output_parsers/json.py
"""

import json
import re


class JsonParser:
    def __init(self):
        """Class to parse json produced by LLMs. Inspiration taken from langchain.
        It fixes quotes, it escapes separators, etc."""
        pass

    @staticmethod
    def _replace_new_line(match: re.Match[str]) -> str:
        value = match.group(2)
        value = re.sub(r"\n", r"\\n", value)
        value = re.sub(r"\r", r"\\r", value)
        value = re.sub(r"\t", r"\\t", value)
        value = re.sub('"', r"\"", value)

        return match.group(1) + value + match.group(3)

    @staticmethod
    def _custom_parser(multiline_string: str) -> str:
        """
        The LLM response for `action_input` may be a multiline
        string containing unescaped newlines, tabs or quotes. This function
        replaces those characters with their escaped counterparts.
        (newlines in JSON must be double-escaped: `\\n`)
        """
        if isinstance(multiline_string, (bytes, bytearray)):
            multiline_string = multiline_string.decode()

        multiline_string = re.sub(
            r'("action_input"\:\s*")(.*)(")',
            JsonParser._replace_new_line,
            multiline_string,
            flags=re.DOTALL,
        )

        return multiline_string

    @staticmethod
    def parse_json(json_string: str) -> dict:
        """
        Parse a JSON string from LLM response

        Args:
            json_string: The Markdown string.

        Returns:
            The parsed JSON object as a Python dictionary.
        """
        json_str = json_string

        # Strip whitespace and newlines from the start and end
        json_str = json_str.strip()

        # handle newlines and other special characters inside the returned value
        json_str = JsonParser._custom_parser(json_str)

        # Parse the JSON string into a Python dictionary
        parsed = json.loads(json_str)

        return parsed
