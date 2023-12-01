"""
This file contains the OpenAI chat API wrappers.
"""
import logging
from typing import Any

import openai
from tenacity import (  # for exponential backoff
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from gears.history import History, Message
from gears.llms.base import BaseLLM

logger = logging.getLogger(__name__)


OPENAI_PRICING_MAP = {
    "gpt-3.5-turbo": {
        "prompt_tokens": float(0.0015 / 1000),
        "completion_tokens": float(0.002 / 1000),
    },
    "gpt-35-turbo": {
        "prompt_tokens": float(0.0015 / 1000),
        "completion_tokens": float(0.002 / 1000),
    },
    "gpt-3.5-turbo-16k": {
        "prompt_tokens": float(0.003 / 1000),
        "completion_tokens": float(0.004 / 1000),
    },
    "gpt-35-turbo-16k": {
        "prompt_tokens": float(0.003 / 1000),
        "completion_tokens": float(0.004 / 1000),
    },
    "gpt-4": {
        "prompt_tokens": float(0.03 / 1000),
        "completion_tokens": float(0.06 / 1000),
    },
    "gpt-4-32k": {
        "prompt_tokens": float(0.06 / 1000),
        "completion_tokens": float(0.12 / 1000),
    },
}


class OpenAIChat(BaseLLM):
    def __init__(
        self, model: str = "gpt-3.5-turbo", max_retries: int = 3, **kwargs: Any
    ):
        """Creates an OpenAI chat API wrapper.

        Args:
            model (str, optional): String representing the name of the model. Defaults to "gpt-3.5-turbo".
            max_retries (int, optional): Max number of retries for calling the OpenAI API. Defaults to 3.
            **kwargs: Any additional kwargs to pass to the OpenAI chat API.
        """
        self.model = model
        self.api_kwargs = kwargs
        self.max_retries = max_retries

        # strip date from model
        split_model = self.model.split("-")
        if len(split_model) >= 4 and "k" not in split_model[-1]:
            # Truncade the last part of the model name
            self.model_base = "-".join(split_model[:-1])
        else:
            self.model_base = self.model

    def retry_policy(self):
        return retry(
            wait=wait_random_exponential(min=1, max=30),
            stop=stop_after_attempt(self.max_retries),
        )

    async def chat_api_call(self, request: dict):
        retry_decorator = self.retry_policy()
        decorated_func = retry_decorator(self._chat_api_call_impl)
        return await decorated_func(request)

    async def _chat_api_call_impl(self, request: dict):
        try:
            response = await openai.ChatCompletion.acreate(**request)
            assert (
                "content" in response["choices"][0]["message"]
            ), "No content in response"
            return response
        except Exception as e:
            logger.error(
                f"Exception when calling OpenAI occurred: {e}. The response was: {response}. History was {request['messages']}."
            )
            raise

    async def run(
        self,
        prompt: str,
        history: History,
        **message_kwargs: Any,
    ) -> Any:
        """Calls the OpenAI chat API with the given prompt and chat history.
        Also adds the response to the chat history and increments the cost.

        Args:
            prompt (str): Prompt to send to the OpenAI chat API.
            history (History): History of the chat so far.
            **message_kwargs (Any): Any additional kwargs to pass to the message.

        Returns:
            Any: Raw response from the OpenAI chat API.
        """

        # Construct chat history
        curr_message = Message(role="user", content=prompt, **message_kwargs)
        history.add(curr_message)
        try:
            messages = [m.model_dump() for m in history]
        except AttributeError:
            messages = [m.dict() for m in history]
        request = {
            "model": self.model,
            "messages": messages,
            **self.api_kwargs,
        }
        response = await self.chat_api_call(request)
        try:
            returned_message = response["choices"][0]["message"]["content"]
            history.add(
                Message(
                    role=response["choices"][0]["message"]["role"],
                    content=returned_message,
                )
            )
        except KeyError:
            logger.error(f"Could not find message and/or role in response: {response}")

        # Increment cost
        try:
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]

            cost = (
                OPENAI_PRICING_MAP[self.model_base]["prompt_tokens"] * prompt_tokens
                + OPENAI_PRICING_MAP[self.model_base]["completion_tokens"]
                * completion_tokens
            )
            history.increment_cost(cost)
        except KeyError:
            logger.error(
                f"Could not find pricing for model {self.model}. Not incrementing cost."
            )

        return response


class AzureOpenAIChat(OpenAIChat):
    def __init__(
        self,
        model: str = "gpt-35-turbo",
        deployment_id: str = "gpt-35-turbo",
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Creates an Azure OpenAI chat API wrapper.

        Args:
            model (str, optional): String representing the name of the model. Defaults to "gpt-3.5-turbo". Used only to determine pricing.
            deployment_id (str, optional): String representing the deployment ID of the model based on your Azure Deployment. Defaults to "gpt-3.5-turbo".
            max_retries (int, optional): Max number of retries for calling the OpenAI API. Defaults to 3.
            **kwargs: Any additional kwargs to pass to the chat API.
        """
        self.deployment_id = deployment_id

        super().__init__(model=model, max_retries=max_retries, **kwargs)

    async def run(
        self,
        prompt: str,
        history: History,
        **message_kwargs: Any,
    ) -> Any:
        """Calls the Azure OpenAI chat API with the given prompt and chat history. Also adds the response to the chat history and increments the cost.

        Uses the `deployment_id` to call the relevant Azure OpenAI chat API.

        Args:
            prompt (str): Prompt to send to the OpenAI chat API.
            history (History): History of the chat so far.
            **message_kwargs (Any): Any additional kwargs to pass to the message.

        Returns:
            Any: Raw response from the OpenAI chat API.
        """
        # Construct chat history
        curr_message = Message(role="user", content=prompt, **message_kwargs)
        history.add(curr_message)
        try:
            messages = [m.model_dump() for m in history]
        except AttributeError:
            messages = [m.dict() for m in history]
        request = {
            "deployment_id": self.deployment_id,
            "messages": messages,
            **self.api_kwargs,
        }
        response = await self.chat_api_call(request)
        try:
            returned_message = response["choices"][0]["message"]["content"]
            history.add(
                Message(
                    role=response["choices"][0]["message"]["role"],
                    content=returned_message,
                )
            )
        except KeyError:
            logger.error(f"Could not find message and/or role in response: {response}")

        # Increment cost
        try:
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]

            cost = (
                OPENAI_PRICING_MAP[self.model_base]["prompt_tokens"] * prompt_tokens
                + OPENAI_PRICING_MAP[self.model_base]["completion_tokens"]
                * completion_tokens
            )
            history.increment_cost(cost)
        except KeyError:
            logger.error(
                f"Could not find pricing for model {self.model}. Not incrementing cost."
            )

        return response
