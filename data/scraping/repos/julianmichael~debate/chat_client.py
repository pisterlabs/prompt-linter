# Adapted from: https://github.com/akbir/debate

import aiohttp
import json

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from prompts import WORD_LIMIT

from fastapi import HTTPException

OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/complete"

class RateLimitError(Exception):
    
    def __init__(self, message="Rate limit exceeded"):
        super().__init__(message)

class ChatClient:

    def __init__(self, model: str, api_key: str, org_key: str, max_context_length: int):
        self.model = model
        self.api_key = api_key
        self.org_key = org_key
        self.max_context_length = max_context_length

    # for exponential backoff
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3), retry=retry_if_exception_type(RateLimitError))
    async def chat_completion_with_backoff_async(self, session, messages, temperature):
        if self.model.startswith("claude"):
            import anthropic
            async with session.post(
                    ANTHROPIC_BASE_URL,
                    headers={
                        "X-API-key": f"{self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Client": anthropic.constants.ANTHROPIC_CLIENT_VERSION,
                    },
                    data=json.dumps({
                        "prompt": f"{anthropic.HUMAN_PROMPT}{messages}{anthropic.AI_PROMPT}",
                        "model": self.model,
                        "max_tokens_to_sample": WORD_LIMIT,
                        "temperature": temperature,
                    }),
            ) as resp:
                if resp.status == 200:
                    response = await resp.json()
                    return response.get("completion")
                else:
                    raise HTTPException(status_code=resp.status, detail=(await resp.json()))
                # elif resp.status == 429:
                #     print("Anthropic API rate limit exceeded")
                #     raise openai.error.OpenAIError()
                # else:
                #     print(f"Error: {resp.status} {await resp.text()}")
                #     raise Exception(f"Error: {resp.status} {await resp.text()}")

        else:
            async with session.post(
                    OPENAI_BASE_URL,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                        # "OpenAI-Organization": f"{self.org_key}",
                    },
                    data=json.dumps({
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                    }),
            ) as resp:
                if resp.status == 200:
                    response = await resp.json()
                    return response["choices"][0]["message"]["content"]
                elif resp.status == 429:
                    raise RateLimitError()
                else:
                    response = await resp.json()
                    message = response['error']['message']
                    raise HTTPException(status_code=resp.status, detail=message)
                # else:
                #     print(f"Error: {resp.status} {await resp.text()}")
                #     raise Exception(f"Error: {resp.status} {await resp.text()}")