from __future__ import annotations

import logging
from typing import Dict, List

from langchain.schema import HumanMessage, SystemMessage

from ..llms.base import ChatLLM
from .util import get_buffer_string, load_yaml_file, parse_conversation


class LanguageFunction:
    """
    Single turn natural language function which expects a text query in a structured format and returns a text response.
    """

    def __init__(self, config: Dict, llm: ChatLLM) -> None:
        """
        Initialize the function
        Args:
            config (Dict): The containing the function's configuration.
        """
        function = dict(config)
        self.system_message = SystemMessage(content=function.get("instruction", ""))
        self.few_shot_prompt = parse_conversation(function.get("examples", []))
        self.user_message_template: str = function["input_template"]
        self.llm = llm

    def __call__(self, **kwargs) -> str:
        """
        Call the Agent Function with the given arguments.

        Args:
            callback (bool): Whether to use the OpenAI callback for logging
            kwargs (Dict): The arguments to the function.
        
        Returns:
            str: The response from the function.
        """
        message = HumanMessage(content=self.user_message_template.format(**kwargs))
        return self._call_model(message)

    def _call_model(self, message: HumanMessage) -> str:
        response = self.llm([self.system_message, *self.few_shot_prompt, message])
        chat_string = get_buffer_string(
            [self.system_message, *self.few_shot_prompt, message, response]
        )
        logging.info(f"Language Function thread:\n{chat_string}")
        return response.content

    @classmethod
    def from_yaml(cls, filepath: str, llm: ChatLLM) -> LanguageFunction:
        """
        Load an agent from a YAML file.

        Args:
            filepath (str): The path to the YAML file.

        Returns:
            LanguageFunction: The configured language function.
        """
        yaml_obj = load_yaml_file(filepath)
        return cls(yaml_obj, llm)
