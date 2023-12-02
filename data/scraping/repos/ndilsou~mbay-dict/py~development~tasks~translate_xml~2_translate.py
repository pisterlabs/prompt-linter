from __future__ import annotations
import asyncio
import datetime as dt
from dataclasses import dataclass
import sys
import time
from typing import Any, Coroutine, cast
from rich.progress import track
from rich import print as rprint
from pathlib import Path
from instructor import patch, OpenAISchema, dsl
import tiktoken

import json
import openai
from openai.error import RateLimitError as OpenAIRateLimitError
from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

patch()

sem = asyncio.Semaphore(1)


async def main(input_filename: str, output_dirname: str):
    input_filepath = Path(input_filename)
    output_filepath = Path(output_dirname) / "french_translated.json"
    checkpoint_filepath = Path(output_dirname) / "french_translated.chk.json"

    # Load the JSON data from the input file
    with input_filepath.open("r") as f:
        data = json.load(f)

    tasks: list[Coroutine[Any, Any, None]] = []

    processor = OpenaiRequestProcessor()
    # processor.start()
    # Iterate over each entry in the data
    try:
        for entry in data:
            # Translate each English field in the entry
            for field in ["translate", "gramnote", "sortfield"]:
                if field in entry:
                    tasks.append(assign_translation(processor, entry, field))

            # Translate each English field in the samples
            for sample in entry.get("samples", []):
                for field in ["trans_sent"]:
                    if field in sample:
                        tasks.append(assign_translation(processor, sample, field))

            # Translate each English field in the expressions
            for expression in entry.get("expressions", []):
                for field in ["trans_exp", "trans_sent"]:
                    if field in expression:
                        tasks.append(assign_translation(processor, expression, field))

        for i, t in enumerate(track(asyncio.as_completed(tasks), total=len(tasks))):
            await t
            if i % 500 == 0:
                # Commit the translated data to the output file
                with checkpoint_filepath.open("w") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

        # Write the translated data to the output file
        with output_filepath.open("w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            rprint("Wrote output file")

        rprint("Done")

    finally:
        pass
        # await processor.stop()


async def assign_translation(
    processor: OpenaiRequestProcessor, item: dict[str, str], field: str
):
    if f"french_{field}" in item:
        # rprint(f"Skipping {field} for {item['entrytranc']}")
        return

    if item.get(field) is None:
        return

    text = item[field]
    # rprint(f"Translating {field} for {item['entrytranc']}")
    async with sem:
        translation = await translate_text(text)
    item[f"french_{field}"] = translation
    await asyncio.sleep(0.01)


class FrenchTranslation(OpenAISchema):
    """ "
    Correctly translated text from English to French
    """

    french_text: str = Field(..., description="A valid high quality French translation")


@retry(wait=wait_random_exponential(min=30, max=90), stop=stop_after_attempt(6))
async def translate_text(text: str) -> str:
    """Translate text from English to French using OpenAI's API"""
    request = Request(text)
    response: FrenchTranslation = await request.completion.acreate()
    return response.french_text


# rprint(translation)

# return translation.french_text


class Request:
    text: str
    retry_count: int
    _response: str | None
    _ready: asyncio.Event
    completion: dsl.ChatCompletion

    def __init__(self, text: str):
        self.text = text
        self.retry_count = 0
        self._response = None
        self._ready = asyncio.Event()
        self.completion = self._completion()
        comp_args = self.completion.kwargs
        self.num_tokens = num_tokens_from_messages(
            [m for m in comp_args["messages"]]
        ) + num_tokens_from_functions(comp_args["functions"])

    def set_response(self, response: str):
        self._response = response
        self._ready.set()

    async def get(self):
        await self._ready.wait()
        if self._response is None:
            raise Exception(f"Response for '{self.text}' is None")

        return self._response

    async def send(self) -> None:
        # async with sem:
        try:
            response = await self.completion.acreate()
        except OpenAIRateLimitError as e:
            raise RateLimitError(f"rate limit hit for '{self.text}'", self, e) from e
        except Exception as e:
            raise TranslateError("request failed", self) from e

        self.set_response(response.french_text)

    def add_retry(self):
        self.retry_count += 1

    def _completion(self) -> dsl.ChatCompletion:
        return cast(
            dsl.ChatCompletion,
            dsl.ChatCompletion(
                model="gpt-4-0613",
                name="french_translation",
                temperature=0,
            )
            | dsl.SystemTask(
                "You are translating from English to French for a linguistics dictionary."
            )
            | dsl.SystemTips(
                [
                    "You may see the following abbreviations: '[s.o.]' => [someone], [s.t.] => something"
                ]
            )
            | dsl.UserMessage(self.text)
            | FrenchTranslation,
        )


class TranslateError(Exception):
    request: Request

    def __init__(self, message: str, request: Request):
        super().__init__(message)
        self.request = request


class RateLimitError(TranslateError):
    oai_error: OpenAIRateLimitError

    def __init__(self, message: str, request: Request, oai_error: OpenAIRateLimitError):
        super().__init__(message, request)
        self.oai_error = oai_error


# account for
# - rate limit
# - not blocking the submit loop


class OpenaiRequestProcessor:
    SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR = 15
    SECONDS_TO_SLEEP_EACH_LOOP = 0.06  # this gives us about 3500 requests per minute

    def __init__(
        self, max_token_per_minute: int = 40000, max_request_per_minute: int = 3300
    ):
        self.stopped = asyncio.Event()
        self.request_queue: asyncio.Queue[Request] = asyncio.Queue()
        self.retry_queue: asyncio.Queue[Request] = asyncio.Queue()
        self.inflight_queue: asyncio.Queue[asyncio.Task[None]] = asyncio.Queue()
        self.max_tokens_per_minute = max_token_per_minute
        self.max_requests_per_minute = max_request_per_minute
        self.available_token_capacity = max_token_per_minute
        self.available_request_capacity = max_request_per_minute
        self.last_update_time = time.time()

    async def translate_text(self, text: str) -> str:
        """Translate text from English to French using OpenAI's API"""
        request = Request(text)
        await self.request_queue.put(request)
        return await request.get()

    async def stop(self):
        self.stopped.set()
        await asyncio.gather(self.submit_loop, self.complete_loop)

    def start(self):
        self.complete_loop = asyncio.create_task(self._complete_loop())
        self.submit_loop = asyncio.create_task(self._submit_loop())
        self.monitor_loop = asyncio.create_task(self._monitor_loop())

    async def _monitor_loop(self):
        while True:
            if self.stopped.is_set():
                break

            rprint(
                f"INFO({dt.datetime.now()}): available_token_capacity: {self.available_token_capacity}, available_request_capacity: {self.available_request_capacity}"
            )
            await asyncio.sleep(15)

    async def _submit_loop(self):
        rprint("DEBUG: Starting submit loop")
        request: Request | None = None
        while True:
            if self.stopped.is_set():
                rprint("Stopping submit loop")
                break

            if request is None:
                try:
                    if not self.retry_queue.empty():
                        request = self.retry_queue.get_nowait()
                        self.retry_queue.task_done()
                    else:
                        request = self.request_queue.get_nowait()
                        self.request_queue.task_done()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(self.SECONDS_TO_SLEEP_EACH_LOOP)
                    continue
                except Exception as e:
                    self.stopped.set()
                    raise

            self._update_available_capacity()
            if (
                self.available_token_capacity < request.num_tokens
                or self.available_request_capacity < 1
            ):
                rprint(
                    f"Rate limit reached. Sleeping for {self.SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR} seconds"
                )
                await asyncio.sleep(self.SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR)
                continue
            else:
                self.available_token_capacity -= request.num_tokens
                self.available_request_capacity -= 1
                await self.inflight_queue.put(asyncio.create_task(request.send()))

            request = None
            await asyncio.sleep(self.SECONDS_TO_SLEEP_EACH_LOOP)

    async def _complete_loop(self):
        rprint("DEBUG: Starting complete loop")
        workers = [asyncio.create_task(self._create_worker()) for _ in range(10)]
        rprint("DEBUG: Created all workers")
        await self.stopped.wait()
        await self.inflight_queue.join()
        for worker in workers:
            worker.cancel()

    async def _create_worker(self):
        rprint("Starting worker")
        try:
            while True:
                try:
                    task = self.inflight_queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)
                    continue
                try:
                    await task
                    task.result()
                except RateLimitError as e:
                    rprint(
                        f"rate limited ({e.request.num_tokens}), retried {e.request.retry_count} times)",
                        e,
                    )
                    e.request.add_retry()
                    await self.retry_queue.put(e.request)
                except TranslateError as e:
                    rprint(e)
                    e.request.add_retry()
                    await self.retry_queue.put(e.request)
                except Exception as e:
                    rprint(e)
                    raise
                finally:
                    self.inflight_queue.task_done()
        except Exception as e:
            rprint("Dead worker.", e)
            raise

    def _update_available_capacity(self):
        current_time = time.time()
        seconds_since_update = current_time - self.last_update_time
        self.available_request_capacity = min(
            self.available_request_capacity
            + self.max_requests_per_minute * seconds_since_update / 60.0,
            self.max_requests_per_minute,
        )
        self.available_token_capacity = min(
            self.available_token_capacity
            + self.max_tokens_per_minute * seconds_since_update / 60.0,
            self.max_tokens_per_minute,
        )
        self.last_update_time = current_time


def num_tokens_from_messages(
    messages: list[dict[str, str]], model="gpt-3.5-turbo-0613"
) -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_tokens_from_functions(
    functions: list[dict[str, Any]], model="gpt-3.5-turbo-0613"
):
    """Return the number of tokens used by a list of functions."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for function in functions:
        function_tokens = len(encoding.encode(function["name"]))
        function_tokens += len(encoding.encode(function["description"]))

        if "parameters" in function:
            parameters = function["parameters"]
            if "properties" in parameters:
                for propertiesKey in parameters["properties"]:
                    function_tokens += len(encoding.encode(propertiesKey))
                    v = parameters["properties"][propertiesKey]
                    for field in v:
                        if field == "type":
                            function_tokens += 2
                            function_tokens += len(encoding.encode(v["type"]))
                        elif field == "description":
                            function_tokens += 2
                            function_tokens += len(encoding.encode(v["description"]))
                        elif field == "enum":
                            function_tokens -= 3
                            for o in v["enum"]:
                                function_tokens += 3
                                function_tokens += len(encoding.encode(o))
                        else:
                            print(f"Warning: not supported field {field}")
                function_tokens += 11

        num_tokens += function_tokens

    num_tokens += 12
    return num_tokens


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
