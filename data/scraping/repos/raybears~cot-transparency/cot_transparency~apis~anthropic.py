import logging
from collections.abc import Sequence

import anthropic
from dotenv import load_dotenv
from retry import retry

from cot_transparency.apis.base import InferenceResponse, ModelCaller, Prompt
from cot_transparency.apis.util import (
    convert_assistant_if_completion_to_assistant,
    messages_has_none_role,
)
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import (
    ChatMessage,
    StrictChatMessage,
    StrictMessageRole,
)
from cot_transparency.util import setup_logger

logger = setup_logger(__name__, logging.INFO)


class AnthropicPrompt(Prompt):
    def __str__(self) -> str:
        return self.format()

    def format(self) -> str:
        if messages_has_none_role(self.messages):
            raise ValueError(f"Anthropic chat messages cannot have a None role. Got {self.messages}")
        messages = convert_assistant_if_completion_to_assistant(self.messages)

        # Add the required empty assistant tag for Claude models if the last message does not have the assistant role
        if messages[-1].role == StrictMessageRole.user:
            messages.append(StrictChatMessage(role=StrictMessageRole.assistant, content=""))

        message = ""
        for msg in messages:
            match msg.role:
                case StrictMessageRole.user:
                    message += f"{anthropic.HUMAN_PROMPT} {msg.content}"
                case StrictMessageRole.assistant:
                    message += f"{anthropic.AI_PROMPT} {msg.content}"
                case StrictMessageRole.none:
                    message += f"\n\n{msg.content}"
                case StrictMessageRole.system:
                    # No need to add something infront for system messages
                    message += f"\n\n{msg.content}"
        return message


class AnthropicCaller(ModelCaller):
    def __init__(self):
        load_dotenv()
        self.client = anthropic.Anthropic()

    @retry(
        exceptions=(anthropic.APIError, anthropic.APIConnectionError),
        tries=3,
        delay=5,
        logger=logger,
    )
    @retry(exceptions=(anthropic.RateLimitError), tries=-1, delay=1, logger=logger)
    def call(
        self,
        messages: Sequence[ChatMessage],
        config: OpenaiInferenceConfig,
    ) -> InferenceResponse:
        assert "claude" in config.model, f"AnthropicCaller can only be used with claude models. Got {config.model}"
        text = AnthropicPrompt(messages=messages).format()

        resp = self.client.completions.create(
            prompt=text,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=config.model,
            max_tokens_to_sample=config.max_tokens,
            temperature=config.temperature,  # type: ignore
        )
        inf_response = InferenceResponse(raw_responses=[resp.completion])
        return inf_response
