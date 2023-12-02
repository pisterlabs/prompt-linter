import os
from dataclasses import dataclass
from enum import Enum

import discord
import openai

SEPARATOR_TOKEN = "<|endoftext|>"
MAX_THREAD_MESSAGES = 50
MAX_CHARS_PER_REPLY_MSG = (
    1500  # discord has a 2k limit, we just break message into 1.5k
)


class CompletionResult(Enum):
    OK = 0
    TOO_LONG = 1
    INVALID_REQUEST = 2
    OTHER_ERROR = 3


@dataclass
class CompletionData:
    status: CompletionResult
    reply_text: str | None
    status_text: str | None


class OpenAIClient:
    def __init__(self):
        self.openai = openai
        self.openai.organization = os.getenv("ZERODAY_TTS_OPENAPI_ORG")
        self.openai.api_key = os.getenv("ZERODAY_TTS_OPENAPI_TOKEN")

    @staticmethod
    def split_into_shorter_messages(message: str) -> list[str]:
        return [
            message[i : i + MAX_CHARS_PER_REPLY_MSG]
            for i in range(0, len(message), MAX_CHARS_PER_REPLY_MSG)
        ]

    def generate_completion_response(self, message: list[dict]) -> CompletionData:
        try:
            response = self.openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
            )
            reply = response["choices"][0]["message"]["content"]
            return CompletionData(
                status=CompletionResult.OK, reply_text=reply, status_text=None
            )
        except Exception as e:
            return CompletionData(
                status=CompletionResult.OTHER_ERROR,
                reply_text=None,
                status_text=str(e),
            )
