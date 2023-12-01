import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from domain.port.openai_port import OpenAIPort


class AskGenerate:
    """
    AskGenerate class
    """

    logging.warning(__name__)

    def execute_service(
        self, question: str, text: str, task: str, port: "OpenAIPort"
    ):
        """
        execute_service cognitive search
        Args: question
        Returns: cognitive search
        """
        return port.ask_openai(question, text, task)
