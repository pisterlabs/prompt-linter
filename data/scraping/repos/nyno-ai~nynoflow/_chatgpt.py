from typing import Optional, Union, cast
from warnings import warn

import openai
from attrs import define, field
from openai.error import ServiceUnavailableError as OpenaiServiceUnavailableError

from nynoflow.chats._chatgpt._chatgpt_objects import (
    ChatgptMessageHistory,
    ChatgptRequestMessage,
    ChatgptResponse,
)
from nynoflow.chats.chat_objects import ChatMessage
from nynoflow.exceptions import ServiceUnavailableError
from nynoflow.tokenizers import OpenAITokenizer
from nynoflow.util import logger


@define
class ChatgptProvider:
    """LLM Engine for OpenAI ChatGPT."""

    api_key: str = field()
    model: str = field()

    n: int = 1  # Number of completions to generate
    stream = False  # Streams not supported in this library yet
    organization: Optional[str] = field(default=None)
    temperature: Optional[float] = field(default=None)
    top_p: Optional[float] = field(default=None)
    stop: Optional[Union[str, list[str]]] = field(default=None)
    max_tokens: Optional[int] = field(default=None)
    presence_penalty: Optional[float] = field(default=None)
    frequency_penalty: Optional[float] = field(default=None)
    logit_bias: Optional[dict[str, float]] = field(default=None)
    user: Optional[str] = field(default=None)

    provider_id: str = field(default="chatgpt")
    openai_chat_completion_client: openai.ChatCompletion = field(init=False)
    tokenizer: OpenAITokenizer = field(init=False)
    retries_on_service_error: int = field(default=0)

    # It is accesible using ChatgptProvider.token_limit using the getter function, and
    # setable using ChatgptProvider(token_limit=n) using the setter function, and by deafult
    # is set to the token limit of the model using the post attributes init function.
    token_limit: int = field()

    @token_limit.default
    def _token_limit_default(self) -> Union[int, None]:
        """Get the token limit for the model.

        Returns:
            Union[int, None]: The token limit.
        """
        return self._model_token_limit()

    def __attrs_post_init__(self) -> None:
        """Configure the openai package with auth and configurations."""
        self.openai_chat_completion_client = openai.ChatCompletion()

        self.tokenizer = OpenAITokenizer(self.model)

    def _model_token_limit(
        self,
    ) -> int:  # pragma: no cover # Ignored because no use testing each model separately
        """Get the token limit for the model.

        Raises:
            NotImplementedError: If the model is not supported.

        Returns:
            int: The token limit.
        """
        # Warn the user if he uses a model without a specific version, and alert him to the model being assumed.
        model_for_token_limit = self.model
        if self.model == "gpt-3.5-turbo":
            warn(
                "Model gpt-3.5-turbo may change in the future, and it is recommended to"
                "use a specific version of the model instead. Using token limit for model gpt-3.5-turbo-0613 instead."
            )
            model_for_token_limit = "gpt-3.5-turbo-0613"  # noqa: S105
        elif self.model == "gpt-4":
            warn(
                "Model gpt-4 may change in the future, and it is recommended to"
                "use a specific version of the model instead. Using token limit for model gpt-4-0613 instead."
            )
            model_for_token_limit = "gpt-4-0613"  # noqa: S105

        # Get the token limit for the model
        if model_for_token_limit in {"gpt-4-32k-0613", "gpt-4-32k-0314"}:
            return 32768

        if model_for_token_limit in {"gpt-4-0314", "gpt-4-0613"}:
            return 8192

        if model_for_token_limit in {"gpt-3.5-turbo-0613", "gpt-3.5-turbo-0301"}:
            return 4096

        if model_for_token_limit in {"gpt-3.5-turbo-16k-0613"}:
            return 16384

        raise NotImplementedError(
            f"Token limit not implemented for model {model_for_token_limit}. Please open an issue on GitHub."
        )

    def _convert_message_history(
        self, message_history: list[ChatMessage]
    ) -> ChatgptMessageHistory:
        """Convert the message history to a ChatGPT format message history.

        Args:
            message_history (list[ChatMessage]): The message history to convert.

        Returns:
            ChatgptMessageHistory: The converted message history.
        """
        return [
            ChatgptRequestMessage(role=msg.role, content=msg.content)
            for msg in message_history
        ]

    def completion(self, messages: list[ChatMessage]) -> str:
        """Get a completion from the OpenAI API.

        Raises:
            ServiceUnavailableError: If the OpenAI API is unavailable.

        Args:
            messages (list[ChatMessage]): The message history to prompt with.

        Returns:
            The completion.
        """
        chatgpt_messages: ChatgptMessageHistory = self._convert_message_history(
            messages
        )
        optional_params = {
            "n": self.n,
            "stream": self.stream,
            "organization": self.organization,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "user": self.user,
        }

        params = {
            "api_key": self.api_key,
            "model": self.model,
            "messages": chatgpt_messages,
            **{k: v for k, v in optional_params.items() if v is not None},
        }
        logger.debug(f"Calling the OpenAI API with params: {params}")
        try:
            res: ChatgptResponse = self.openai_chat_completion_client.create(**params)
        except OpenaiServiceUnavailableError as err:
            raise ServiceUnavailableError from err

        content = cast(str, res["choices"][0]["message"]["content"])
        return content
