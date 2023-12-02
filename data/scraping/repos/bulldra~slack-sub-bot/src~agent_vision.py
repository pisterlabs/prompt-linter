"""This is a agent that uses the GPT-4 Vision model to answer questions about images."""
from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

import agent_gpt
import slack_link_utils


class AgentVision(agent_gpt.AgentGPT):
    """
    This is a agent that uses the GPT-4 Vision model to answer questions about image.
    """

    def __init__(
        self, context: dict[str, Any], chat_history: list[dict[str, str]]
    ) -> None:
        """初期化"""
        super().__init__(context, chat_history)
        self._openai_model = "gpt-4-vision-preview"
        self._openai_stream = False

    def build_prompt(
        self, chat_history: list[dict[str, Any]]
    ) -> list[
        ChatCompletionSystemMessageParam
        | ChatCompletionUserMessageParam
        | ChatCompletionAssistantMessageParam
        | ChatCompletionToolMessageParam
        | ChatCompletionFunctionMessageParam
    ]:
        url: str = slack_link_utils.extract_and_remove_tracking_url(
            self._chat_history[-1]["content"]
        )

        prompt_messages: list[
            ChatCompletionSystemMessageParam
            | ChatCompletionUserMessageParam
            | ChatCompletionAssistantMessageParam
            | ChatCompletionToolMessageParam
            | ChatCompletionFunctionMessageParam
        ] = [
            ChatCompletionUserMessageParam(
                role="user",
                content=[
                    {"type": "text", "text": "このイメージの説明をしてください。"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url,
                            "detail": "high",
                        },
                    },
                ],
            )
        ]
        return prompt_messages
