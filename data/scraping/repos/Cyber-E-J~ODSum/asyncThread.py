"""Tools to generate from OpenAI prompts."""

import asyncio
import logging
import os
from typing import Any

import aiolimiter
import openai
import openai.error
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
import random


async def _throttled_openai_chat_completion_acreate(
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        limiter: aiolimiter.AsyncLimiter,
        timeout: int = 60
) -> dict[str, Any]:
    async with limiter:
        for _ in range(10):
            try:
                return await asyncio.wait_for(  # 修改这里
                    openai.ChatCompletion.acreate(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    ), timeout  # 添加这里
                )
            except asyncio.TimeoutError:  # 修改这里，注意去掉 or openai.error.Timeout
                logging.warning(f"TimeoutError after 60 seconds. Returning default value.")
                return {"choices": [{"message": {"content": "Timeout, returning default value"}}]}  # 返回一个默认结果
            except openai.error.RateLimitError as e:
                error_message = str(e)
                if 'You exceeded your current quota' in error_message:
                    print(error_message)
                    return {"choices": [{"message": {"content": ""}}]}
                logging.warning(
                    f"OpenAI RateLimitError: {e}. Sleeping for 11 seconds.Try {_ + 1}"
                )
                await asyncio.sleep(11)
            except openai.error.Timeout as e:
                logging.warning(f"OpenAI Timeout: {e}. Sleeping for 11 seconds.Try {_ + 1}")
                await asyncio.sleep(11)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}.Sleeping for 11 seconds.Try {_ + 1}")
                await asyncio.sleep(11)
            except openai.error.ServiceUnavailableError as e:
                logging.warning(f"OpenAI ServiceUnavailableError:{e}Try {_ + 1}")
                await asyncio.sleep(11)
            except Exception as e:
                logging.warning(f"Exception OR Error:{e}Try {_ + 1},sleep 11 sec")
                await asyncio.sleep(11)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
        api_key: str,
        messages,
        engine_name: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        requests_per_minute: int,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    openai.api_key = api_key
    session = ClientSession()
    openai.aiosession.set(session)
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    await session.close()
    return [x["choices"][0]["message"]["content"] for x in responses]


def run(messages, engine_name, temperature, max_tokens, top_p, api_key, requests_per_minute):
    # messages = [[{"role": "user", "content": f"print the number {i}"}] for i in range(100)]
    responses = asyncio.run(generate_from_openai_chat_completion(
        messages=messages,
        engine_name=engine_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        api_key=api_key,
        requests_per_minute=requests_per_minute
    ))
    return responses
