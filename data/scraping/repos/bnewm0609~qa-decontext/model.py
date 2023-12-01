import json
import os
import time
from typing import List, Dict, Union, Optional

import anthropic
import openai
import tiktoken

from decontext.cache import DiskCache, CacheState
from decontext.data_types import (
    OpenAIChatMessage,
    OpenAIChatResponse,
    OpenAICompletionResponse,
    AnthropicResponse,
    ModelResponse,
)
from decontext.logging import warn

OPENAI_CHAT_MODEL_NAMES = {
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
}


class GPT3Model:
    """Training and predicting with GPT3.

    Attributes:
        _name: the name of the model.
        cache: A local cache to avoid sending the same request twice.
        params: The default generation parameters sent to the model with the prompt.
        is_chat_model: True if the model requires hitting the OpenAI Chat endpoint.
        is_anthropic_model: True if the model is Claude.
    """

    def __init__(self, model_name, **params) -> None:
        """Initialize GPT3 model.

        This involves setting up the cache, API key, and default parameters.

        Args:
            args (DictConfig): Experiment configuration arguments.
        """
        self._name = model_name
        self.cache = DiskCache.load()
        if "OPENAI_API_KEY" not in os.environ:
            warn(
                "OPENAI_API_KEY not found in environment variables."
                "Set OPEN_API_KEY with your API key to use the OpenAI API."
            )
        else:
            openai.api_key = os.environ["OPENAI_API_KEY"]

        self.params = {
            "model": self.name,
            "logprobs": 5,
            "user": "[ANON]",
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 1.0),
            "max_tokens": params.get("max_tokens", 150),
            "stop": ["END"],
        }

        self.is_chat_model = self.name in OPENAI_CHAT_MODEL_NAMES
        self.is_anthropic_model = False

    @property
    def name(self):
        """Getter for the name attribute"""

        return self._name

    @name.setter
    def name(self, value):
        """Write the value to _name and the params list."""

        self._name = value
        self.params["model"] = value

    def calculate_cost(
        self,
        prompt: Optional[Union[str, List[OpenAIChatMessage]]] = None,
        completion: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        max_gen_len: Optional[int] = None,
    ) -> float:
        """Use the latest prices to estimate the price of calling the given model on the passed dataset.

        Note: Depending on the parameters passed, it might only calculate the cost of the prompt, not the
        generation.

        Args:
            model_name: which model - determines the price.
            dataset: dataset - determines the number of examples and price per example.
            prompt_tokens: Number of tokens in the prompt. If specified, it's used to calculate the price instead
                of the dataset. It can be passed along with other arguments.
            completion_tokens: Number of tokens in the completion. This is separate from `prompt_tokens` because
                some OpenAI models charge differently for tokens in the prompt vs. those in the completion.
            max_gen_len: if specified, is added to the number of tokens estimated using the
                        dataset. Is not used if only prompt_tokens/completion_tokens are specified but not dataset.

        Returns:
            A float with the price of calling the model on the dataset.
        """
        price_per_1k_input_token_map: Dict[str, float] = {
            "ada": 0.0004,
            "babbage": 0.0005,
            "curie": 0.002,
            "davinci": 0.02,
            "code-davinci-002": 0.0,
            "gpt-3.5-turbo": 0.0015,
            "gpt-3.5-turbo-0301": 0.0015,
            "gpt-3.5-turbo-0613": 0.0015,
            "gpt-3.5-turbo-16k": 0.003,
            "gpt-3.5-turbo-16k-0613": 0.003,
            "gpt-4": 0.03,
            "gpt-4-0314": 0.03,
            "gpt-4-0613": 0.03,
            "text-ada-001": 0.0004,
            "text-babbage-001": 0.0005,
            "text-curie-001": 0.002,
            "text-davinci-001": 0.02,
            "text-davinci-002": 0.02,
            "text-davinci-003": 0.02,
            "claude-instant-1": 1.63 / 1_000,
            "claude-1": 11.02 / 1_000,
            "claude-2": 11.02 / 1_000,
        }

        price_per_1k_output_token_map: Dict[str, float] = {  # type: ignore
            **price_per_1k_input_token_map,
            "gpt-3.5-turbo-0301": 0.002,
            "gpt-3.5-turbo": 0.002,
            "gpt-3.5-turbo-0613": 0.002,
            "gpt-3.5-turbo-16k": 0.004,
            "gpt-3.5-turbo-16k-0613": 0.004,
            "gpt-4": 0.06,
            "gpt-4-0314": 0.06,
            "gpt-4-0613": 0.06,
            "claude-instant-1": 5.51 / 1_000,
            "claude-1": 32.68 / 1_000,
            "claude-2": 32.68 / 1_000,
        }

        price_per_1k_output = price_per_1k_output_token_map[self.name]
        price_per_1k_input = price_per_1k_input_token_map[self.name]

        using_token_counts = prompt_tokens is not None and completion_tokens is not None
        using_tokens = (
            prompt is not None and (completion is not None or max_gen_len is not None) and not using_token_counts
        )

        if not using_token_counts and not using_tokens:
            raise ValueError(
                "Either specify token counts (using `prompt_tokens` and `completion_tokens`)"
                "or strings that have to be tokenized using `prompt` and `completion` or `prompt` and"
                " `max_gen_len`"
            )

        if prompt_tokens is None and prompt is not None:
            if self.is_anthropic_model:
                prompt_tokens = anthropic.count_tokens(prompt)
            else:
                tokenizer = tiktoken.encoding_for_model(self.name)
                if self.is_chat_model and isinstance(prompt, list):
                    prompt_tokens = sum([len(tokenizer.encode(message.content)) for message in prompt])
                else:
                    prompt_tokens = len(tokenizer.encode(prompt))

            # if only the prompt is provided, estimate the number of tokensi n the output by assuming that
            # max_gen_len tokens are always geneated. Note: this is an overestimate
            if completion_tokens is None and completion is None:
                completion_tokens = max_gen_len
            elif completion is not None:
                if self.is_anthropic_model:
                    completion_tokens = anthropic.count_tokens(completion)
                else:
                    completion_tokens = len(tokenizer.encode(completion))

        # input_price = prompt_tokens * price_per_1k_input / 1_000
        # output_price = completion_tokens * price_per_1k_output / 1_000
        if prompt_tokens is None or completion_tokens is None:
            raise ValueError(
                "(2) Either specify token counts (using `prompt_tokens` and `completion_tokens`)"
                "or strings that have to be tokenized using `prompt` and `completion` or `prompt` and"
                " `max_gen_len`"
            )

        total_price = (prompt_tokens * price_per_1k_input + completion_tokens * price_per_1k_output) / 1_000
        return total_price

    def extract_text(self, response):
        if self.is_anthropic_model:
            return response.completion
        elif self.is_chat_model:
            return response.choices[0].message.content
        else:
            return response.choices[0].text

    def get_key(self, params: dict) -> str:
        """Creates a dict that is serialized to a json string"""
        _key = {k: v for k, v in params.items() if k not in {"user", "prompt", "messages"}}
        if self.is_chat_model:
            _key["messages"] = [m for m in params["messages"]]
        else:
            _key["prompt"] = params["prompt"]
        return json.dumps(_key, sort_keys=True)  # sort keys for consistent serialization

    def prompt_with_cache(self, params, cache_state: Optional[CacheState] = None):
        """Send a request to the API with the given params if they haven't been used yet.

        This is done by creating a unique key based on the params dict and having the cache handle running
        the function to prompt the model if the key is not in the cache.

        Args:
            params (dict): The parameters used to prompt the model with.
        """

        key = self.get_key(params)

        def prompt():
            # GPT4 has a lower rate-limit.
            if "gpt-4" in self.name:
                time.sleep(0.25)
            else:
                time.sleep(0.1)
            try:
                if self.is_chat_model:
                    response = openai.ChatCompletion.create(**params)
                else:
                    response = openai.Completion.create(**params)
            except openai.error.InvalidRequestError:
                print("Stopping to investigate why there was an invalid request to the API...")
                breakpoint()
            return response.to_dict_recursive()

        return self.cache.query(key, prompt, cache_state=cache_state)

    def __call__(self, text_prompt: Union[str, List[OpenAIChatMessage]], cache_state=None) -> ModelResponse:
        """Perform inference on the model with the given prompt.

        Overwrite the params with the given prompt. For Chat models, use a simple system message and put the
        prompt in the user message."""
        params = {k: v for k, v in self.params.items()}
        params["prompt"] = text_prompt

        response = self.prompt_with_cache(params, cache_state=cache_state)

        if self.is_anthropic_model:
            result = AnthropicResponse.parse_obj(response)
            result.cost = self.calculate_cost(
                prompt=text_prompt,
                completion=result.completion,
            )
        else:
            result = OpenAICompletionResponse.parse_obj(response)
            result.cost = self.calculate_cost(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
            )

        return result


class GPT3ChatModel(GPT3Model):
    """Run inference with the Chat endpoint and arbitrary messages."""

    def __init__(self, model_name, **params) -> None:
        super().__init__(model_name, **params)
        self.chat_model = True

    def __call__(
        self, messages_prompt: Union[str, List[OpenAIChatMessage]], cache_state: Optional[CacheState] = None
    ) -> Union[OpenAIChatResponse, OpenAICompletionResponse]:  # type: ignore[override]
        params = {k: v for k, v in self.params.items()}
        params.pop("logprobs")
        if isinstance(messages_prompt, str):
            params["messages"] = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant",
                },
                {"role": "user", "content": messages_prompt},
            ]
        else:
            params["messages"] = [message.dict() for message in messages_prompt]

        response = self.prompt_with_cache(params, cache_state=cache_state)
        result = OpenAIChatResponse.parse_obj(response)
        result.cost = self.calculate_cost(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
        )
        return result


class ClaudeModel(GPT3Model):
    """Call the Anthropic Claude API."""

    def __init__(self, model_name: str, **params) -> None:
        """Initailize the model.

        Create an anthropic client and use a smaller set of parameters compared to OpenAI.
        """

        self._name = model_name
        self.cache = DiskCache.load()
        self.client = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])
        self.params = {
            "stop_sequences": [anthropic.HUMAN_PROMPT],
            "model": self.name,
            "max_tokens_to_sample": params.get("max_tokens", 150),
        }

        # Technically it is a chat model, but we're querying it like it's a completion model.
        self.is_chat_model = False
        self.is_anthropic_model = True

    def prompt_with_cache(self, params, cache_state: Optional[CacheState] = None):
        """Prompt with the anthropic library instead of the OpenAI one."""
        key = "-".join([f"{param_k}_{param_v}" for param_k, param_v in params.items()])

        key += f"-prompt_{params['prompt']}"  # [:100]  # that should be enough, right?

        def prompt():
            time.sleep(0.1)
            try:
                response = self.client.completion(**params)
            except anthropic.ApiException:
                print("Stopping so you can determine why there was an API exception.")
                breakpoint()
            return response

        return self.cache.query(key, prompt, cache_state=cache_state)


def load_model(model_name: str, **params) -> GPT3Model:
    if model_name in OPENAI_CHAT_MODEL_NAMES:
        return GPT3ChatModel(model_name, **params)
    elif "claude" in model_name:
        return ClaudeModel(model_name, **params)
    else:
        return GPT3Model(model_name, **params)
