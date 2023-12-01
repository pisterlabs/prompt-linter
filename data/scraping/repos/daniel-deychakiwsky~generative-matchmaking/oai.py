import base64
import copy
import os
from typing import Any, Collection, Dict, List, Set

import openai
from openai import ChatCompletion, Image

openai.api_key = os.getenv("OPENAI_API_KEY")


class ConversationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"ConversationError: {message}")


class Conversation:
    # message keys
    M_R: str = "role"
    M_C: str = "content"
    # role names
    R_S: str = "system"
    R_U: str = "user"
    R_A: str = "assistant"
    # role turns
    R_TURN_MAP: Dict[str, str] = {R_S: R_U, R_U: R_A, R_A: R_U}
    R_INIT_SET: Set[str] = {R_S, R_U}

    def __init__(self) -> None:
        self.conversation: List = []

    def add_system_message(self, message: str) -> None:
        message = message.strip()
        if message:
            self.conversation.append({self.M_R: self.R_S, self.M_C: message})

    def add_user_message(self, message: str) -> None:
        message = message.strip()
        if message:
            self.conversation.append({self.M_R: self.R_U, self.M_C: message})

    def add_assistant_message(self, message: str) -> None:
        message = message.strip()
        if message:
            self.conversation.append({self.M_R: self.R_A, self.M_C: message})

    def get_messages(self) -> List[Dict[str, str]]:
        self._validate_turns()
        self._validate_messages()
        return copy.deepcopy(self.conversation)

    def _validate_turns(self) -> None:
        if not self.conversation:
            raise ConversationError("empty")

        role = self.conversation[0][self.M_R]

        if role not in self.R_INIT_SET:
            raise ConversationError("invalid")

        for m in self.conversation[1:]:
            if m[self.M_R] != self.R_TURN_MAP[role]:
                raise ConversationError("invalid")
            role = m[self.M_R]

    def _validate_messages(self) -> None:
        if not self.conversation:
            raise ConversationError("empty")

        for m in self.conversation:
            if not m[self.M_C]:
                raise ConversationError("invalid")


# https://platform.openai.com/docs/api-reference/chat/create
def chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    max_tokens: int = 10,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    functions: list[dict[str, Collection[str]]] | None = None,
    function_call: Dict[str, str] | None = None,
) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }

    # https://platform.openai.com/docs/guides/gpt/function-calling
    if functions and function_call:
        kwargs["functions"] = functions
        kwargs["function_call"] = function_call
        func_completion: str = (
            ChatCompletion.create(**kwargs).choices[0].message.function_call.arguments
        )
        return func_completion
    else:
        completion: str = ChatCompletion.create(**kwargs).choices[0].message.content
        return completion


def text_to_image(
    prompt: str,
    response_format: str = "b64_json",
    n_images: int = 1,
    size: str = "256x256",
) -> bytes:
    truncated_prompt: str = prompt[:397] + "..." if len(prompt) > 400 else prompt

    response: Image = Image.create(
        prompt=truncated_prompt,
        response_format=response_format,
        n=n_images,
        size=size,
    )

    img_data: str = response["data"][0]["b64_json"]
    img_bytes: bytes = base64.decodebytes(img_data.encode("utf-8"))

    return img_bytes
