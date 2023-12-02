import os
from typing import Generator

import httpx
import openai_async

from conf import OPEN_AI_API_KEY


def tokens_to_chunks(
    tokens: list[int], chunk_size: int = 3000, overlap: int = 100
) -> Generator[list[int], None, None]:
    num_tokens = len(tokens)
    for i in range(0, num_tokens, chunk_size - overlap):
        chunk = tokens[i : i + chunk_size]
        yield chunk


async def send_partial_text(text: str, target_doc: str) -> str:
    messages = [
        {"role": "system", "content": f"This is text summarization for {target_doc}"},
        {"role": "user", "content": f"Summarize this documentation: {text}"},
    ]
    try:
        response = await openai_async.chat_complete(
            OPEN_AI_API_KEY,
            timeout=60,
            payload={
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 500,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 1,
            },
        )
        res = response.json()
        return res["choices"][0]["message"]["content"].strip()
    except httpx.TimeoutException as e:
        raise RuntimeError(f"Timeout while summarizing {target_pep}")


async def summarize(link: str, text: str) -> str:
    prompt_request = f"Please summarize the text with bullet points: {text}"
    try:
        response = await openai_async.complete(
            OPEN_AI_API_KEY,
            timeout=60,
            payload={
                "model": "text-davinci-003",
                "prompt": prompt_request,
                "temperature": 0.3,
                "max_tokens": 150,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 1,
            },
        )
        res = response.json()
        content = res["choices"][0]["text"].strip()
        summary = f"{content}\n\nFor more information: {link}"
        return summary
    except httpx.TimeoutException as e:
        raise RuntimeError(f"Timeout while summarizing {link}")
