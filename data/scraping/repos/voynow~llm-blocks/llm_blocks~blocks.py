import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import dotenv
import openai

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


def set_api_key(api_key: str):
    openai.api_key = api_key


@dataclass
class OpenAIConfig:
    model_name: str = "gpt-3.5-turbo-16k"
    temperature: float = 0.1
    stream: bool = False


class MessageHandler:
    def __init__(self, system_message: Optional[str] = "You are a helpful assistant."):
        self.system_message = system_message
        self.initialize_messages()

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def initialize_messages(self):
        self.messages = []
        self.add_message("system", self.system_message)


class CompletionHandler(Protocol):
    def create_completion(self, block: "Block") -> Any:
        ...

    def parse_message(self, message: Any) -> str:
        ...


class StreamCompletionHandler(CompletionHandler):
    def create_completion(self, block: "Block") -> Any:
        return openai.ChatCompletion.create(
            model=block.config.model_name,
            messages=block.message_handler.messages,
            temperature=block.config.temperature,
            stream=True,
        )

    def parse_message(self, message: Any) -> str:
        full_response_content = ""
        for msg in message:
            delta = msg["choices"][0]["delta"]
            parsed_content = delta["content"] if "content" in delta else ""
            full_response_content += parsed_content
        return full_response_content


class BatchCompletionHandler(CompletionHandler):
    def create_completion(self, block: "Block") -> Any:
        return openai.ChatCompletion.create(
            model=block.config.model_name,
            messages=block.message_handler.messages,
            temperature=block.config.temperature,
            stream=False,
        )

    def parse_message(self, message: Any) -> str:
        return message["choices"][0]["message"]["content"]


class Block:
    def __init__(
        self,
        config: OpenAIConfig,
        message_handler: MessageHandler,
        completion_handler: CompletionHandler,
    ):
        self.config = config
        self.message_handler = message_handler
        self.completion_handler = completion_handler

    def prepare_content(self, content: str) -> str:
        return content

    def execute(self, content: str) -> Optional[str]:
        self.message_handler.initialize_messages()
        prepared_content = self.prepare_content(content)
        self.message_handler.add_message("user", prepared_content)
        message = self.completion_handler.create_completion(self)
        return self.completion_handler.parse_message(message)

    def __call__(self, content: str) -> Optional[str]:
        return self.execute(content)


class TemplateBlock(Block):
    def __init__(self, template: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.template = template
        self.input_variables = re.findall(r"\{(\w+)\}", self.template)

    def execution_prep(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        inputs = {}
        if args:
            inputs = {key: value for key, value in zip(self.input_variables, args)}
        if kwargs:
            inputs.update(kwargs)
        return self.template.format(**inputs)

    def execute(self, *args: Any, **kwargs: Any) -> Optional[str]:
        return super().execute(self.execution_prep(*args, **kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[str]:
        return self.execute(*args, **kwargs)


class ChatBlock(Block):
    def execute(self, content: str) -> Optional[str]:
        self.message_handler.add_message("user", content)

        message = self.completion_handler.create_completion(self)
        response = self.completion_handler.parse_message(message)

        self.message_handler.add_message("assistant", response)

        return response
