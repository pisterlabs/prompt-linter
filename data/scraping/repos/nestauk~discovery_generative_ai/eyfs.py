import asyncio
import json
import logging
import string

from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import aiofiles
import openai

from openai.error import APIConnectionError
from openai.error import APIError
from openai.error import RateLimitError
from openai.error import ServiceUnavailableError
from openai.error import Timeout
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from genai import MessageTemplate


logger = logging.getLogger(__name__)


class TextGenerator:
    """Generate tokens using OpenAI's API."""

    @classmethod
    def generate(
        cls,
        messages: List[Union[str, Dict]],
        message_kwargs: Optional[Dict] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        **openai_kwargs,
    ) -> Union[Dict, str]:
        """Generate text using OpenAI's API.

        More details on the API and messages: https://platform.openai.com/docs/guides/gpt/chat-completions-api

        Args:
            messages
                A list of messages to send to the API. They can be:
                - dictionaries
                - str (JSON file path)
                - instances of classes that inherit from BasePromptTemplate

            message_kwargs
                A dictionary of keyword arguments to pass to the messages.

            model
                The OpenAI model to use.

            temperature
                The sampling temperature.

            openai_kwargs
                Keyword arguments to pass to the OpenAI API.

        Returns:
            A dictionary containing the response from the API.

        """
        if not message_kwargs:
            message_kwargs = {}

        messages = [cls.prepare_message(message, **message_kwargs) for message in messages]

        response = cls._call(
            messages=messages,
            temperature=temperature,
            model=model,
            **openai_kwargs,
        )

        return response

    @classmethod
    def prepare_message(cls, obj: Union[MessageTemplate, dict, str], **kwargs) -> Dict:
        """Process a message."""
        if not isinstance(obj, MessageTemplate):
            prompt = MessageTemplate.load(obj)
        else:
            prompt = obj

        prompt.format_message(**kwargs)

        return prompt.to_prompt()

    @staticmethod
    @retry(
        retry(
            reraise=True,
            stop=stop_after_attempt(6),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=(
                retry_if_exception_type(Timeout)
                | retry_if_exception_type(APIError)
                | retry_if_exception_type(APIConnectionError)
                | retry_if_exception_type(RateLimitError)
                | retry_if_exception_type(ServiceUnavailableError)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
    )
    def _call(
        messages: List[Dict],
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict:
        response = openai.ChatCompletion.create(
            messages=messages,
            temperature=temperature,
            **kwargs,
        )

        return response  # type: ignore

    @staticmethod
    def _extract_placeholders(s: str) -> List[str]:
        """Extract placeholder variables that can be filled in an f-string."""
        formatter = string.Formatter()
        return [field_name for _, field_name, _, _ in formatter.parse(s) if field_name is not None]


class EYFSClassifier:
    """Classify text to EYFS areas of learning."""

    @classmethod
    def generate(
        cls,
        messages: List[Union[str, Dict]],
        message_kwargs: Optional[Dict] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        **openai_kwargs,
    ) -> Union[Dict, str]:
        """Generate text using OpenAI's API.

        More details on the API and messages: https://platform.openai.com/docs/guides/gpt/chat-completions-api

        Args:
            messages
                A list of messages to send to the API. They can be:
                - dictionaries
                - str (JSON file path)
                - instances of classes that inherit from BasePromptTemplate

            message_kwargs
                A dictionary of keyword arguments to pass to the messages.

            model
                The OpenAI model to use.

            temperature
                The sampling temperature.

            openai_kwargs
                Keyword arguments to pass to the OpenAI API.

        Returns:
            A dictionary containing the response from the API.

        """
        if not message_kwargs:
            message_kwargs = {}

        messages = [cls.prepare_message(message, **message_kwargs) for message in messages]

        response = cls._call(
            messages=messages,
            temperature=temperature,
            model=model,
            **openai_kwargs,
        )

        parsed_response = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])
        if parsed_response:
            parsed_response["url"] = message_kwargs["url"]
            return parsed_response

        return message_kwargs["url"]

    @classmethod
    def prepare_message(cls, obj: Union[MessageTemplate, dict, str], **kwargs) -> Dict:
        """Process a message."""
        if not isinstance(obj, MessageTemplate):
            prompt = MessageTemplate.load(obj)
        else:
            prompt = obj

        prompt.format_message(**kwargs)

        return prompt.to_prompt()

    @staticmethod
    @retry(
        retry(
            reraise=True,
            stop=stop_after_attempt(6),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=(
                retry_if_exception_type(Timeout)  # noqa: W503
                | retry_if_exception_type(APIError)  # noqa: W503
                | retry_if_exception_type(APIConnectionError)  # noqa: W503
                | retry_if_exception_type(RateLimitError)  # noqa: W503
                | retry_if_exception_type(ServiceUnavailableError)  # noqa: W503
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
    )
    def _call(
        messages: List[Dict],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict:
        response = openai.ChatCompletion.create(
            messages=messages,
            model=model,
            temperature=temperature,
            **kwargs,
        )

        return response  # type: ignore

    @classmethod
    async def agenerate(
        cls,
        messages: List[Union[str, Dict]],
        message_kwargs: Optional[Dict] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        concurrency: int = 10,
        **openai_kwargs,
    ) -> Dict:
        """Generate text using async OpenAI's API.

        More details on the API and messages: https://platform.openai.com/docs/guides/gpt/chat-completions-api

        Args:
            messages
                A list of messages to send to the API. They can be:
                - dictionaries
                - str (JSON file path)

            message_kwargs
                A dictionary of keyword arguments to pass to the messages.

            temperature
                The sampling temperature.

            openai_kwargs
                Keyword arguments to pass to the OpenAI API.

            concurrency:
                The number of concurrent requests to make.

        Returns:
            A dictionary containing the response from the API.

        """
        semaphore = asyncio.Semaphore(concurrency)
        async with semaphore:
            if not message_kwargs:
                message_kwargs = {}

            messages = [cls.prepare_message(message, **message_kwargs) for message in messages]

            response = await cls._acall(
                messages=messages,
                temperature=temperature,
                model=model,
                **openai_kwargs,
            )

            response = response["choices"][0]["message"]["function_call"]["arguments"]
            parsed_response = await cls._parse_json(response)
            if parsed_response:
                parsed_response["url"] = message_kwargs["url"]
                return parsed_response

            return message_kwargs["url"]

    @staticmethod
    @retry(
        retry(
            reraise=True,
            stop=stop_after_attempt(6),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=(
                retry_if_exception_type(Timeout)  # noqa: W503
                | retry_if_exception_type(APIError)  # noqa: W503
                | retry_if_exception_type(APIConnectionError)  # noqa: W503
                | retry_if_exception_type(RateLimitError)  # noqa: W503
                | retry_if_exception_type(ServiceUnavailableError)  # noqa: W503
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
    )
    async def _acall(
        messages: List[Dict],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict:
        response = await openai.ChatCompletion.acreate(
            messages=messages,
            model=model,
            temperature=temperature,
            **kwargs,
        )

        return response  # type: ignore

    @staticmethod
    async def _try_parse_json(item: str) -> Union[dict, None]:
        try:
            return json.loads(item)
        except json.JSONDecodeError as e:
            return e

    @staticmethod
    async def _parse_json(item: str) -> Union[dict, None]:
        result = await EYFSClassifier._try_parse_json(item)
        if isinstance(result, json.JSONDecodeError):
            result = await EYFSClassifier._try_parse_json(item.replace("'", '"'))
            if isinstance(result, json.JSONDecodeError):
                logging.error(f"Invalid JSON: Error: {str(result)}")
                return None
        return result

    @staticmethod
    async def write_line_to_file(item: dict, filename: str) -> None:
        """Write the item to a file."""
        file = f"{filename}/invalid_json.txt"
        if isinstance(item, dict):
            file = f"{filename}/parsed_json.jsonl"

        async with aiofiles.open(file, "a") as f:
            await f.write(f"{json.dumps(item)}\n")


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """Encode text with OpenAI's text embedding model."""
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]
