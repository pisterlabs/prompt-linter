import asyncio
import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections import ChainMap, defaultdict
from enum import Enum, auto
from functools import cache
from itertools import cycle
from typing import Any, Callable, Coroutine, Literal, Optional, TypeVar

import attrs
import openai
from anthropic import AsyncAnthropic
from anthropic.types.completion import Completion as AnthropicCompletion
from attrs.validators import min_len
from openai.openai_object import OpenAIObject
from openai.openai_object import OpenAIObject as OpenAICompletion
from termcolor import cprint
from tqdm import tqdm

from textsteg.caching import add_to_cache, get_cache

Role = Literal["user", "system", "assistant"]


class StopReason(Enum):
    MAX_TOKENS = auto()
    STOP_SEQUENCE = auto()

    @classmethod
    def factory(cls, stop_reason: str) -> "StopReason":
        """
        Parses the openai and anthropic stop reasons into a StopReason enum.
        """
        if stop_reason in ["max_tokens", "length"]:
            return cls.MAX_TOKENS
        elif stop_reason in ["stop_sequence", "stop"]:
            return cls.STOP_SEQUENCE
        raise ValueError(f"Invalid stop reason: {stop_reason}")

    def __repr__(self):
        return self.name


@attrs.frozen()
class ChatResponse:
    model_id: str
    completion: str
    stop_reason: StopReason = attrs.field(converter=StopReason.factory)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChatResponse":
        return cls(d["model_id"], d["completion"], d["stop_reason"])

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "completion": self.completion, "stop_reason": self.stop_reason.name.lower()}


@attrs.define()
class ChatModel(ABC):
    defaults: dict[str, Any] = attrs.Factory(
        dict
    )  # Default parameters to pass to the API (for example you might want to always set temperature = .8)
    print_msgs: bool = False  # If set to true, will print out messages that are sent to the API.

    @abstractmethod
    async def __call__(self, messages, **kwargs) -> list[ChatResponse]:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


async def call_until_valid(
    chat_model: ChatModel,
    prompt: Any,
    is_valid: Callable[[str], bool],
    num_tries: int = 15,
    num_desired: int = 1,
    **kwargs,
) -> list[ChatResponse]:
    """
    Calls the chat model until `num_desired` valid answers are returned, or until num_tries is reached.
    """
    logging.info(f"Calling call_until_valid for {type(chat_model)}")
    if isinstance(chat_model, OpenAIChatModel):
        assert "n" not in kwargs, "set `n` via `num_desired`, not via kwargs"

    tries = 0
    valid_responses = []
    while (num_needed := num_desired - len(valid_responses)) > 0 and tries < num_tries:
        if tries > 0:
            logging.info(f"Try #{tries}")
        if isinstance(chat_model, OpenAIChatModel):
            responses = await chat_model(prompt, n=num_needed, **kwargs)
        elif isinstance(chat_model, AnthropicChatModel):
            responses = await chat_model(prompt, **kwargs)
        else:
            raise RuntimeError(f"`call_until_valid` not supported for Chat Model of type: {type(chat_model)}")

        valid_responses.extend([response for response in responses if is_valid(response.completion)])
        tries += len(responses)
    if len(valid_responses) != num_desired:
        logging.warn(f"Only {len(valid_responses)} valid responses found after {num_tries} tries, raising RuntimeError")
        raise RuntimeError(f"Invalid formatting encountered after {num_tries} tries")
    return valid_responses


@attrs.define()
class Resource:
    """
    A resource that is consumed over time and replenished at a constant rate.
    """

    refresh_rate: float = attrs.field()  # How many units of the resource are replenished per minute
    value: float = attrs.field(init=False)
    last_update_time: float = attrs.field(init=False, factory=time.time)

    def __attrs_post_init__(self):
        self.value = self.refresh_rate

    def _refresh(self):
        """
        Updates the value of the resource based on the time since the last update.
        """
        curr_time = time.time()
        self.value = min(self.refresh_rate, self.value + (curr_time - self.last_update_time) * self.refresh_rate / 60)
        self.last_update_time = curr_time

    def geq(self, amount: float) -> bool:
        self._refresh()
        return self.value >= amount

    def consume(self, amount: float):
        """
        Consumes the given amount of the resource.
        """
        assert self.geq(amount), f"Resource does not have enough capacity to consume {amount} units"
        self.value -= amount


deep_infra_data = {
    "tokens_per_minute": 500_000,
    "requests_per_minute": 6000,
    "price_per_context_token": 0.0015 / 1000,
    "price_per_completion_token": 0.002 / 1000,
    "context_length": 4096,
    "versions": [None],
    "api_base": "https://api.deepinfra.com/v1/openai",
    "api_key": os.environ.get("DEEPINFRA_API_KEY", None),
}

openai_data = {
    "gpt-4": {
        "tokens_per_minute": 150_000,
        "requests_per_minute": 1000,
        "price_per_context_token": 0.03 / 1000,
        "price_per_completion_token": 0.06 / 1000,
        "context_length": 8192,
        "versions": ["0314", "0613"],
    },
    "gpt-3.5-turbo-16k": {
        "tokens_per_minute": 180_000,
        "requests_per_minute": 3500,
        "price_per_context_token": 0.003 / 1000,
        "price_per_completion_token": 0.004 / 1000,
        "context_length": 16384,
        "versions": ["0613"],
    },
    "gpt-3.5-turbo": {
        "tokens_per_minute": 500_000,
        "requests_per_minute": 6000,
        "price_per_context_token": 0.0015 / 1000,
        "price_per_completion_token": 0.002 / 1000,
        "context_length": 4096,
        "versions": ["0301", "0613"],
    },
    "text-davinci-003": {
        "tokens_per_minute": 250_000,
        "requests_per_minute": 3000,
        "price_per_context_token": 0.02 / 1000,
        "price_per_completion_token": 0.02 / 1000,
        "context_length": 4096,
        "versions": [None],
        "is_completion": True,
    },
    "meta-llama/Llama-2-70b-chat-hf": deep_infra_data,
    "meta-llama/Llama-2-7b-chat-hf": deep_infra_data,
    "codellama/CodeLlama-34b-Instruct-hf": deep_infra_data,
}


def openai_parse_names(model_ids: list[str]) -> list[tuple[str, str]]:
    """
    Parses a list of model ids into a list of tuples of (base_model_id, release_date).
    """
    names = []
    base_names = openai_data.keys()
    for model_id in model_ids:
        if "/" in model_id or model_id.startswith("text-davinci"):
            assert model_id in openai_data, f"Invalid model id: {model_id}"
            names.append((model_id, None))
            continue
        if model_id.startswith("ft"):
            model_id = model_id.split(":")[1]
        if match := re.match(r"^(?P<base_name>gpt-(?:4|3.5-turbo)(?:-16k)?)(?:-(?P<release_date>\d+))?$", model_id):
            base_name = match.group("base_name")
            release_date = match.group("release_date")
            if base_name not in base_names:
                raise ValueError(f"Invalid model id: {model_id}")
            if release_date is not None and release_date not in openai_data[base_name]["versions"]:
                raise ValueError(f"Invalid model id: {model_id}")
            names.append((base_name, release_date))
        else:
            raise ValueError(f"Invalid model id: {model_id}")
    return names


@attrs.define()
class OpenAIChatModel(ChatModel):
    """
    Wrapper around OpenAI API that handles rate limiting for you.
    If you pass in more than one model_id, the cheapest model that has capacity will be selected.
    """

    model_ids: list[str] = attrs.field(
        default=None, validator=[min_len(1), lambda _self, _attribute, value: len(set(value)) == len(value)]
    )
    max_attempts: int = 11  # Max number of times to retry if the API fails
    buffer_factor: float = 0.98  # Set to a number a bit smaller than 1.

    # rate limit
    token_capacity: dict[str, Resource] = attrs.field(init=False, default=attrs.Factory(dict))
    request_capacity: dict[str, Resource] = attrs.field(init=False, default=attrs.Factory(dict))

    # For calculating costs
    context_token_price: dict[str, float] = attrs.field(init=False, default=attrs.Factory(dict))
    completion_token_price: dict[str, float] = attrs.field(init=False, default=attrs.Factory(dict))
    num_context_tokens: dict[str, int] = attrs.field(init=False, default=attrs.Factory(lambda: defaultdict(int)))
    num_completion_tokens: dict[str, int] = attrs.field(init=False, default=attrs.Factory(lambda: defaultdict(int)))

    context_lengths: dict[str, int] = attrs.field(init=False, default=attrs.Factory(dict))
    is_completions: dict[str, bool] = attrs.field(init=False, default=attrs.Factory(dict))
    api_bases: dict[str, str] = attrs.field(init=False, default=attrs.Factory(dict))
    api_keys: dict[str, str] = attrs.field(init=False, default=attrs.Factory(dict))

    def __attrs_post_init__(self):
        for model_id, (base_id, _version) in zip(self.model_ids, openai_parse_names(self.model_ids)):
            self.token_capacity[model_id] = Resource(openai_data[base_id]["tokens_per_minute"] * self.buffer_factor)
            self.request_capacity[model_id] = Resource(openai_data[base_id]["requests_per_minute"] * self.buffer_factor)
            self.context_token_price[model_id] = openai_data[base_id]["price_per_context_token"]
            self.completion_token_price[model_id] = openai_data[base_id]["price_per_completion_token"]
            self.context_lengths[model_id] = openai_data[base_id]["context_length"]
            self.is_completions[model_id] = openai_data[base_id].get("is_completion", False)
            self.api_bases[model_id] = openai_data[base_id].get("api_base", None)
            self.api_keys[model_id] = openai_data[base_id].get("api_key", None)

        # sort model_id by price so we always try to use the cheapest model
        self.model_ids.sort(key=lambda model_id: self.completion_token_price[model_id])

    @staticmethod
    def _to_openai_messages(messages: list[tuple[Role, str]]) -> list[dict[str, str]]:
        return [{"role": speaker, "content": text} for speaker, text in messages]

    @staticmethod
    def _to_chat_response(completion_response: OpenAICompletion) -> OpenAICompletion:
        completion_response["choices"] = [
            OpenAIObject.construct_from(
                {
                    "index": m["index"],
                    "finish_reason": m["finish_reason"],
                    "message": {"role": "assistant", "content": m["text"]},
                }
            )
            for m in completion_response["choices"]
        ]
        return completion_response

    def _calc_token_count(self, messages, params):
        # The magic formula is: .25 * (total number of characters) + (number of messages) + (max_tokens, or 15 if not specified)
        BUFFER = 5  # A bit of buffer for some error margin

        num_tokens = 0
        for message in messages:
            num_tokens += 1
            num_tokens += len(message[1]) / 4
        return params.get("n", 1) * (int(num_tokens + BUFFER) + params.get("max_tokens", 15))

    async def __call__(
        self, messages: list[tuple[Role, str]], call_timeout: Optional[float] = None, **kwargs
    ) -> list[ChatResponse]:
        oaimsgs = self._to_openai_messages(messages)
        params = ChainMap(kwargs, self.defaults)
        token_count = self._calc_token_count(messages, params)
        if token_count > max(self.context_lengths.values()):
            raise ValueError(f"Token count ({token_count}) exceeds max context length ({max(self.context_lengths)})")

        async def attempt_api_call():
            for model_id in cycle(self.model_ids):
                request_capacity, token_capacity = self.request_capacity[model_id], self.token_capacity[model_id]
                long_enough_context = token_count <= self.context_lengths[model_id]
                if request_capacity.geq(1) and token_capacity.geq(token_count) and long_enough_context:
                    request_capacity.consume(1)
                    token_capacity.consume(token_count)

                    if self.is_completions[model_id]:
                        used_params = {
                            **params,
                            # max_tokens=None default to 16
                            "max_tokens": params.get("max_tokens") or (1024 + token_count),
                        }
                        api_response: OpenAICompletion = await openai.Completion.acreate(
                            api_base=self.api_bases[model_id],
                            api_key=self.api_keys[model_id],
                            prompt="\n\n".join([msg["content"] for msg in oaimsgs]) + "\n\n",
                            engine=model_id,
                            stream=False,
                            **used_params,
                        )  # type: ignore
                        api_response = self._to_chat_response(api_response)
                    else:
                        api_response: OpenAICompletion = await openai.ChatCompletion.acreate(
                            api_base=self.api_bases[model_id],
                            api_key=self.api_keys[model_id],
                            messages=oaimsgs,
                            model=model_id,
                            stream=False,
                            **params,
                        )  # type: ignore

                    self.num_context_tokens[model_id] += api_response.usage.prompt_tokens
                    self.num_completion_tokens[model_id] += api_response.usage.completion_tokens
                    return [
                        ChatResponse(model_id, choice.message.content, choice.finish_reason)
                        for choice in api_response.choices
                    ]
                else:
                    await asyncio.sleep(0.01)

        def _get_color(role: Role) -> str:
            return {"user": "cyan", "system": "magenta", "assistant": "light_green"}[role]

        if self.print_msgs:
            for role, text in messages:
                cprint(f"{role.upper()}:", "white")
                cprint(text, _get_color(role))

        responses: Optional[list[ChatResponse]] = None
        o_call_timeout = call_timeout
        for i in range(self.max_attempts):
            try:
                if call_timeout is not None and i > 1:
                    call_timeout = o_call_timeout + 1.5 ** (i - 1)
                responses = await asyncio.wait_for(
                    attempt_api_call(),
                    timeout=call_timeout,
                )
            except Exception as e:
                logging.warn(f"Encountered API {type(e).__name__}: {str(e)}.\nRetrying now. (Attempt {i})")
                if isinstance(e, openai.InvalidRequestError):
                    raise e
                await asyncio.sleep(1.5**i * random.uniform(1, 1.1))
                # raise e
            else:
                break

        if responses is None:
            raise RuntimeError(f"Failed to get a response from the API after {self.max_attempts} attempts.")

        if self.print_msgs:
            for i, response in enumerate(responses):
                cprint(f"RESPONSE {i + 1}:", "white")
                cprint(response.completion, _get_color("assistant"), attrs=["bold"])

        return responses

    def reset_cost(self):
        self.num_context_tokens = defaultdict(int)
        self.num_completion_tokens = defaultdict(int)

    def get_cost(self):
        ans = 0
        for model_id in self.model_ids:
            ans += self.num_context_tokens[model_id] * self.context_token_price[model_id]
            ans += self.num_completion_tokens[model_id] * self.completion_token_price[model_id]
        return ans


ANTHROPIC_RATE_LIMIT = 5


@attrs.define()
class AnthropicChatModel(ChatModel):
    model_id: str = attrs.field(default=None, validator=attrs.validators.in_(["claude-2", "claude-instant-1"]))
    max_attempts: int = 5  # Max number of times to retry if the API fails

    # rate limits
    available_requests: asyncio.BoundedSemaphore = attrs.field(
        init=False, default=asyncio.BoundedSemaphore(ANTHROPIC_RATE_LIMIT)
    )

    # For calculating costs
    num_context_tokens: int = attrs.field(init=False, default=0)
    num_completion_tokens: int = attrs.field(init=False, default=0)
    cost_per_context_token: float = attrs.field(init=False)
    cost_per_completion_token: float = attrs.field(init=False)

    # client
    client: AsyncAnthropic = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.cost_per_context_token = {"claude-instant-1": 1.63 / 1_000_000, "claude-2": 11.02 / 1_000_000}[
            self.model_id
        ]
        self.cost_per_completion_token = {"claude-instant-1": 5.51 / 1_000_000, "claude-2": 32.68 / 1000}[self.model_id]
        self.client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    async def __call__(self, messages: str, **kwargs) -> list[ChatResponse]:
        def _get_color(role: str) -> str:
            return {"human": "cyan", "assistant": "light_green"}[role]

        params = ChainMap(kwargs, self.defaults)

        if self.print_msgs:
            pattern = r"\n\n((Human: |Assistant: ).*)"
            for match in re.finditer(pattern, messages):
                cprint(match.group(1), _get_color(match.group(2).removesuffix(": ").lower()))

        response: Optional[AnthropicCompletion] = None
        for i in range(self.max_attempts):
            try:
                async with self.available_requests:
                    response = await self.client.completions.create(prompt=messages, model=self.model_id, **params)

                self.num_context_tokens += await self.client.count_tokens(messages)
                self.num_context_tokens += await self.client.count_tokens(response.completion)  # type: ignore
            except Exception as e:
                logging.warn(f"{self.model_id} encountered API error: {str(e)}")
                if isinstance(e, openai.InvalidRequestError):
                    raise e

                logging.warn(f"Retrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)
                # raise e
            else:
                break

        if response is None:
            raise RuntimeError(f"Failed to get a response from the API after {self.max_attempts} attempts.")

        if self.print_msgs:
            cprint("Response:", "white")
            cprint(f"{response.completion}", _get_color("assistant"), attrs=["bold"])

        return [ChatResponse(self.model_id, response.completion, response.stop_reason)]

    def reset_cost(self):
        self.num_context_tokens = 0
        self.num_completion_tokens = 0

    def get_cost(self):
        return (
            self.cost_per_context_token * self.num_context_tokens
            + self.cost_per_completion_token * self.num_completion_tokens
        )


T = TypeVar("T")


def as_sync(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    import asyncio

    import nest_asyncio

    nest_asyncio.apply()

    def _wrapper(*args, **kwargs) -> T:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(func(*args, **kwargs))

    return _wrapper


def check_fmt(prompts: list[list[tuple[Role, str]]]):
    assert isinstance(prompts, list)
    assert all(isinstance(prompt, list) for prompt in prompts)
    assert all(isinstance(message, tuple) for prompt in prompts for message in prompt)
    assert all(len(message) == 2 for prompt in prompts for message in prompt)
    assert all(role in ["user", "system", "assistant"] for prompt in prompts for role, _ in prompt)
    assert all(isinstance(text, str) for prompt in prompts for _, text in prompt)


@as_sync
async def run_llm(
    llm: OpenAIChatModel,
    prompts: list[list[tuple[Role, str]]],
    desc: str = "",
    cached: bool = True,  # cache the whole batch, not individual prompts
    temperature: float = 0.0,
    call_timeout: Optional[float] = None,
    cache_name: str = "irun_llm",
    **generation_kwargs,
) -> list[list[ChatResponse]]:
    check_fmt(prompts)

    if cached:
        keys = [str((llm.model_ids, prompt, temperature, generation_kwargs)) for prompt in prompts]
        to_run = [k not in get_cache(cache_name) for k in keys]
        to_run_prompts = [prompt for prompt, run in zip(prompts, to_run) if run]
    else:
        to_run_prompts = prompts

    pbar = tqdm(total=len(to_run_prompts), desc=desc)
    n_errors_and_success = [0, 0]

    async def run_prompt(prompt: list[tuple[Role, str]]):
        try:
            response = await llm(prompt, temperature=temperature, call_timeout=call_timeout, **generation_kwargs)
        except Exception as e:
            # raise e
            # print(f"Error: {type(e).__name__}: {e}")
            response = []
        pbar.update(1)
        n_errors_and_success[bool(response)] += 1
        pbar.set_postfix(errors=n_errors_and_success[0], success=n_errors_and_success[1])
        return response

    responses = await asyncio.gather(*[run_prompt(prompt) for prompt in to_run_prompts])
    pbar.close()

    if cached:
        reversed_responses = responses[::-1]
        responses = []
        to_add = []
        for k, is_new in zip(keys, to_run):
            if is_new:
                response = reversed_responses.pop()
                to_add.append((k, [r.to_dict() for r in response]))
                responses.append(response)
            else:
                reponse = get_cache(cache_name)[k]
                responses.append([ChatResponse.from_dict(d) for d in reponse])

        assert len(reversed_responses) == 0, f"{len(reversed_responses)=} != 0"
        assert len(responses) == len(prompts), f"{len(responses)=} != {len(prompts)=}"

        for k, v in to_add:
            add_to_cache(cache_name, k, v)

    return responses


@cache
def gpt_3_5s():
    return OpenAIChatModel(model_ids=["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613"])


@cache
def gpt_3_5():
    return OpenAIChatModel(model_ids=["gpt-3.5-turbo"])


@cache
def old_gpt_3_5():
    return OpenAIChatModel(model_ids=["gpt-3.5-turbo-0301"])


@cache
def gpt_4():
    return OpenAIChatModel(model_ids=["gpt-4"])


@cache
def old_gpt_4():
    return OpenAIChatModel(model_ids=["gpt-4-0314"])


@cache
def llama2_chat(size: int):
    if size not in [7, 70]:
        raise ValueError(f"Invalid Llama model size: {size}")
    return OpenAIChatModel(model_ids=[f"meta-llama/Llama-2-{size}b-chat-hf"])


# Demo of how to call the models
@as_sync
async def openai_demo():
    openai_model = OpenAIChatModel(model_ids=["gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"])

    openai_messages = [
        [("system", "You are a comedic pirate."), ("user", "Hello!")],
        [("system", "You are a swashbuckling space-faring voyager."), ("user", "Hello!")],
    ]

    answer = await asyncio.gather(*[openai_model(message, n=6, max_tokens=16_000) for message in openai_messages])
    cprint(f"Price: ${openai_model.get_cost():.4f}", "light_cyan")
    return answer


@as_sync
async def anthropic_demo():
    anthropic_model = AnthropicChatModel(model_id="claude-instant-1")
    anthropic_message = "\n\nHuman: Tell me a very creative joke.\n\nAssistant:"
    answer = await asyncio.gather(*[anthropic_model(anthropic_message, max_tokens_to_sample=50) for _ in range(3)])
    cprint(f"Price: ${anthropic_model.get_cost():.4f}", "light_cyan")
    return answer


if __name__ == "__main__":
    # print(
    #     run_llm(
    #         gpt_3_5s(),
    #         [[("system", "You are a comedic pirate."), ("user", "Count to 200")]] * 2,
    #         desc="Paraphrasing",
    #         cached=False,
    #     )
    # )
    print(
        run_llm(
            gpt_3_5s(),
            [[("user", "Count to 10!")], [("user", "Count to 20!")]],
            desc="Paraphrasing",
            cached=True,
        )
    )
    print(
        run_llm(
            gpt_3_5s(),
            [[("user", "Count to 20!")], [("user", "Count to 30!")]],
            desc="Paraphrasing",
            cached=True,
        )
    )

    # def display(nested_list: list[list]):
    #     for i, answer in enumerate(list(chain.from_iterable(nested_list))):
    #         if i > 0:
    #             print("-" * 10)
    #         print(answer.model_id, answer.completion.strip())

    # if "OPENAI_API_KEY" in os.environ:
    #     cprint("OpenAI Demo", "light_cyan")
    #     display(openai_demo())
    # else:
    #     cprint("OpenAI API key not found. Skipping OpenAI demo.", "light_cyan")

    # if "ANTHROPIC_API_KEY" in os.environ:
    #     cprint("Anthropic Demo", "light_cyan")
    #     display(anthropic_demo())
    # else:
    #     cprint("Anthropic API key not found. Skipping Anthropic demo.", "light_cyan")
