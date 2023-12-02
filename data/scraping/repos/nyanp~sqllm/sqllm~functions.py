import json
from functools import lru_cache

from openai import OpenAI


def _simple_call(
    client: OpenAI,
    system_message: str | None,
    source_text: str,
    model: str = "gpt-3.5-turbo",
) -> str:
    messages = [{"role": "user", "content": source_text}]

    if system_message:
        messages.insert(0, {"role": "system", "content": system_message})

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return chat_completion.choices[0].message.content


def _classification_call(
    client: OpenAI,
    system_message: str | None,
    source_text: str,
    categories: list[str],
    model: str = "gpt-3.5-turbo",
) -> str:
    messages = [{"role": "user", "content": source_text}]

    if system_message:
        messages.insert(0, {"role": "system", "content": system_message})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "prediction",
                "description": "result of classification",
                "parameters": {
                    "type": "object",
                    "properties": {"result": {"type": "string", "enum": categories}},
                    "required": ["result"],
                },
            },
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "prediction"}},
    )
    args = chat_completion.choices[0].message.tool_calls[0].function.arguments
    return str(json.loads(args)["result"])


@lru_cache
def sentiment(src: str) -> str:
    client = OpenAI()
    return _classification_call(
        client,
        "Classify the sentiment expressed in the following text. The output should be one of 'positive', 'negative' or 'neutral'.",
        src,
        ["positive", "negative", "neutral"],
    )


@lru_cache
def summarize(src: str) -> str:
    client = OpenAI()
    return _simple_call(
        client,
        "Generate a short summary of a given text. Rely strictly on the provided text, without using any external knowledge.",
        src,
    )


@lru_cache
def ai(src: str) -> str:
    client = OpenAI()
    return _simple_call(client, None, src)
