"""
This is a vanilla example of using the base openai API.
"""

from typing import Generator, Optional

import openai
from pydantic import BaseModel


class ChatCompletionDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    delta: ChatCompletionDelta
    finish_reason: Optional[str] = None
    index: int


class ChatCompletion(BaseModel):
    choices: list[ChatCompletionChoice]
    created: int
    id: str
    model: str
    object: str


def igenerate(prompt: str) -> Generator[str, None, None]:
    """
    Generates a stream of chat messages, given a prompt.

    Args:
        prompt: The single prompt to input to the llm.

    Yields:
        A stream of chat messages content from the llm.
    """
    response: Generator[ChatCompletion, None, None] = openai.ChatCompletion.create(  # type: ignore
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True,
    )

    # map response to ChatCompletion objects
    completion: ChatCompletion
    for completion in map(ChatCompletion.parse_obj, response):
        delta = completion.choices[0].delta

        if delta.content is not None:
            yield delta.content


for message in igenerate("Hello, how is your day today?"):
    print(message, end="")
