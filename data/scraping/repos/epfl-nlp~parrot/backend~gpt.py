from typing import List
import os
import openai
from dataclasses import dataclass
import time

from database import MessageRole, Chat, Message
from app import app

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL_TYPES = ["text_completion", "chat_completion"]


@dataclass
class GPTResponse:
    content: str
    model_args: dict
    usage: dict


class GPTError(Exception):
    pass


def check_model_args(model_args: dict) -> dict:
    if not model_args:
        model_args = {}

    temperature = model_args.get(
        "temperature", app.config["DEFAULT_MODEL_ARGS"]["temperature"]
    )
    if temperature < 0 or temperature > 2:
        raise GPTError("`temperature` must be between 0 and 1")

    top_p = model_args.get("top_p", app.config["DEFAULT_MODEL_ARGS"]["top_p"])
    if top_p < 0 or top_p > 1:
        raise GPTError("`top_p` must be between 0 and 1")

    max_tokens = model_args.get(
        "max_tokens", app.config["DEFAULT_MODEL_ARGS"]["max_tokens"]
    )
    if max_tokens is not None and max_tokens < 0:
        raise GPTError("`max_tokens` must be greater than 0")

    presence_penalty = model_args.get(
        "presence_penalty", app.config["DEFAULT_MODEL_ARGS"]["presence_penalty"]
    )
    if presence_penalty < -2 or presence_penalty > 2:
        raise GPTError("`presence_penalty` must be between -2 and 2")

    frequency_penalty = model_args.get(
        "frequency_penalty", app.config["DEFAULT_MODEL_ARGS"]["frequency_penalty"]
    )
    if frequency_penalty < -2 or frequency_penalty > 2:
        raise GPTError("`frequency_penalty` must be between -2 and 2")

    best_of = model_args.get("best_of", app.config["DEFAULT_MODEL_ARGS"]["best_of"])
    if best_of < 1 or best_of > 3:
        raise GPTError("`best_of` must be between 1 and 3")

    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "best_of": best_of,
    }


def get_prefix(chat: Chat, role: MessageRole) -> str:
    if role == MessageRole.SYSTEM:
        return chat.instruction_prefix
    elif role == MessageRole.USER:
        return chat.user_prefix
    elif role == MessageRole.ASSISTANT:
        return chat.assistant_prefix
    else:
        raise GPTError("Invalid role")


def ask_gpt3(
    chat: Chat,
    messages: List[Message],
    model: str = app.config["DEFAULT_TEXT_COMPLETION_MODEL"],
    model_args: dict = None,
) -> GPTResponse:
    prompt = "\n".join(
        [f"{get_prefix(chat, message.role)}{message.content}" for message in messages]
    )
    prompt = prompt + "\n" + chat.assistant_prefix

    # app.logger.debug(f"Prompt:\n{prompt}")

    start_time = time.time()
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        n=1,
        best_of=model_args.get("best_of", app.config["DEFAULT_MODEL_ARGS"]["best_of"]),
        temperature=model_args.get(
            "temperature", app.config["DEFAULT_MODEL_ARGS"]["temperature"]
        ),
        top_p=model_args.get("top_p", app.config["DEFAULT_MODEL_ARGS"]["top_p"]),
        max_tokens=model_args.get(
            "max_tokens", app.config["DEFAULT_MODEL_ARGS"]["max_tokens"]
        ),
        presence_penalty=model_args.get(
            "presence_penalty", app.config["DEFAULT_MODEL_ARGS"]["presence_penalty"]
        ),
        frequency_penalty=model_args.get(
            "frequency_penalty", app.config["DEFAULT_MODEL_ARGS"]["frequency_penalty"]
        ),
    )
    end_time = time.time()

    app.logger.info(f"GPT-3 Completion time: {end_time - start_time} seconds")

    return GPTResponse(
        content=response["choices"][0]["text"],
        model_args=model_args,
        usage=response["usage"],
    )


def ask_chatgpt(
    chat: Chat,
    messages: List[Message],
    model: dict = app.config["DEFAULT_CHAT_COMPLETION_MODEL"],
    model_args: dict = None,
) -> GPTResponse:
    chat_messages = [
        {"role": str(message.role), "content": message.content} for message in messages
    ]

    start_time = time.time()
    response = openai.ChatCompletion.create(
        model=model,
        messages=chat_messages,
        n=1,
        temperature=model_args.get(
            "temperature", app.config["DEFAULT_MODEL_ARGS"]["temperature"]
        ),
        top_p=model_args.get("top_p", app.config["DEFAULT_MODEL_ARGS"]["top_p"]),
        max_tokens=model_args.get(
            "max_tokens", app.config["DEFAULT_MODEL_ARGS"]["max_tokens"]
        ),
        presence_penalty=model_args.get(
            "presence_penalty", app.config["DEFAULT_MODEL_ARGS"]["presence_penalty"]
        ),
        frequency_penalty=model_args.get(
            "frequency_penalty", app.config["DEFAULT_MODEL_ARGS"]["frequency_penalty"]
        ),
    )
    end_time = time.time()

    app.logger.info(f"ChatGPT Completion time: {end_time - start_time} seconds")

    return GPTResponse(
        content=response["choices"][0]["message"]["content"],
        model_args=model_args,
        usage=response["usage"],
    )
