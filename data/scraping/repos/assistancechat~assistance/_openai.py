# Copyright (C) 2023 Assistance.Chat contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import pathlib
import time

import aiofiles
import openai
from asyncache import cached
from cachetools import LRUCache
from tenacity import (
    retry,
    retry_all,
    retry_if_exception_type,
    retry_if_not_exception_message,
    stop_after_attempt,
    wait_random_exponential,
)

from assistance import _ctx
from assistance._logging import log_info

from ._paths import COMPLETIONS, get_completion_cache_path
from ._utilities import get_hash_digest


async def get_completion_only(**kwargs) -> str:
    response = await _completion_with_back_off(**kwargs)

    stripped_response = response["choices"][0]["message"]["content"].strip()  # type: ignore

    log_info(kwargs["scope"], f"Response: {stripped_response}")

    return stripped_response


# Cache this with an LRU cache as well
async def _completion_with_back_off(**kwargs):
    scope: str = kwargs["scope"]
    del kwargs["scope"]

    assert "scope" not in kwargs

    kwargs_for_cache_hash = kwargs.copy()
    del kwargs_for_cache_hash["api_key"]

    completion_request = json.dumps(kwargs_for_cache_hash, indent=2, sort_keys=True)
    completion_request_hash = get_hash_digest(completion_request)
    completion_cache_path = get_completion_cache_path(
        completion_request_hash, create_parent=True
    )

    try:
        async with aiofiles.open(completion_cache_path, "r") as f:
            return json.loads(await f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    log_info(scope, _ctx.pp.pformat(kwargs_for_cache_hash))

    query_timestamp = time.time_ns()

    response = await _run_completion(kwargs)

    log_info(scope, f"Completion result: {response}")

    asyncio.create_task(_store_cache(completion_cache_path, response))

    return response


@retry(
    retry=retry_all(
        retry_if_not_exception_message("Model maximum reached"),
        retry_if_exception_type(),
    ),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(12),
)
async def _run_completion(kwargs):
    response = await _chat_completion_wrapper(**kwargs)

    return response


async def _chat_completion_wrapper(**kwargs):
    prompt = kwargs["prompt"]
    messages = [{"role": "user", "content": prompt}]

    del kwargs["prompt"]
    kwargs["messages"] = messages

    kwargs["model"] = kwargs["engine"]
    del kwargs["engine"]

    try:
        response = await openai.ChatCompletion.acreate(**kwargs)
    except Exception as e:
        if "This model's maximum context length is" in str(e):
            raise ValueError("Model maximum reached")

        raise

    return response


async def _store_cache(completion_cache_path: pathlib.Path, response):
    async with aiofiles.open(completion_cache_path, "w") as f:
        await f.write(json.dumps(response, indent=2))


async def get_embedding(block: str, api_key) -> list[float]:
    result = await _get_embedding_with_cache(block, api_key)
    return result["data"][0]["embedding"]  # type: ignore


async def _get_embedding_with_cache(block: str, api_key):
    block_hash = get_hash_digest(block)
    cache_path = get_completion_cache_path(block_hash, create_parent=True)

    try:
        async with aiofiles.open(cache_path, "r") as f:
            return json.loads(await f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    logging.info("A new embedding: %s", block)

    result = await _get_embedding(block, api_key)

    asyncio.create_task(_store_cache(cache_path, result))

    return result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(12))
async def _get_embedding(block: str, api_key):
    result = await openai.Embedding.acreate(
        input=block, api_key=api_key, model="text-embedding-ada-002"
    )
    return result
