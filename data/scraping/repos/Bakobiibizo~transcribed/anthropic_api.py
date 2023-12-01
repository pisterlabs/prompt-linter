import os
from typing import Literal, List, Dict, Optional, Any
import loguru
from pydantic import BaseModel
from anthropic import (
    Anthropic,
    HUMAN_PROMPT,
    AI_PROMPT,
    AsyncAnthropic,
    APIStatusError,
    APIConnectionError,
)

from dotenv import load_dotenv
from enum import Enum

logger = loguru.logger

load_dotenv()


# ensure there is a .env file in your root dir with this line:
# ANTHROPIC_API_KEY="sk-xxx-xxx-xxx-xxxxxxxx..."
def get_api_key() -> str:
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        return api_key
    else:
        raise EnvironmentError(f"{logger.exception('ANTHROPIC_API_KEY not found')}")


class Prompt(BaseModel):
    role: str
    content: str


# this is an enum not a base class, they are handled differently in type annotating
class ClaudeOptions(Enum):
    CLAUDE_1 = "claude-instant-1.2"
    CLAUDE_2 = "claude-2.0"


class AnthropicChatBot:
    def __init__(self):
        api_key: str = get_api_key()
        self.claude = Anthropic(auth_token=api_key)
        self.async_claude = AsyncAnthropic(auth_token=api_key)
        self.converter: PromptConverter = PromptConverter()
        # TODO: test token tracker
        self.tokens = AnthropicTokens

    # added both async and regular creates for flexibility
    def generate_prompt(
        self, prompt: str, model: ClaudeOptions, max_tokens_to_sample=None
    ):
        if not prompt:
            raise ValueError(f"{logger.info('Prompt cannot be empty')}")
        if not model:
            model = ClaudeOptions.CLAUDE_2
        if not max_tokens_to_sample:
            max_tokens_to_sample = 90000
        response = None
        try:
            if response := self.claude.completions.create(
                max_tokens_to_sample=90000,
                model=str(model),
                prompt=prompt,
            ):
                return response.completion
            else:
                raise ValueError(f"{logger.info('No response')}")
        except ValueError as error:
            return f"Error: {str(error)}"
        except APIStatusError as error:
            logger.exception(
                f"Caught API status error with response body: {error.response}, status code: {error.status_code}"
            )
        except APIConnectionError as error:
            logger.exception(
                f"Caught API connection error with response body: {error.__cause__}"
            )

    async def async_stream(
        self,
        prompt: str,
        model: ClaudeOptions,
        max_tokens_to_sample: Optional[int],
        stream: Optional[bool],
    ):
        if not prompt:
            raise ValueError(f"{logger.info('Prompt cannot be empty')}")
        if not model:
            model = ClaudeOptions.CLAUDE_2
        if not max_tokens_to_sample:
            max_tokens_to_sample = 90000
        if not stream:
            stream = True
        try:
            completion_stream = await self.async_claude.completions.create(
                prompt=prompt,
                max_tokens_to_sample=max_tokens_to_sample,
                model=str(model),
                stream=stream,
            )
            full_completion = None
            async for completion in completion_stream:
                print(completion.completion, end="", flush=True, file=full_completion)
                return full_completion
        except APIStatusError as error:
            logger.exception(
                f"Caught API status error with response body: {error.response}, status code: {error.status_code}"
            )
        except APIConnectionError as error:
            logger.exception(
                f"Caught API connection error with response body: {error.__cause__}"
            )

    async def async_create(
        self,
        prompt: str,
        model: ClaudeOptions,
        max_tokens_to_sample: int,
        stream: bool,
    ):
        return await self.async_stream(
            prompt=prompt,
            max_tokens_to_sample=max_tokens_to_sample,
            model=model,
            stream=stream,
        )


class PromptConverter:
    def __init__(self):
        # added this atribute for prompt conversions in future integrations
        self.create_prompt = Prompt
        self.human_prompt: Literal["\n\nHuman:"] = HUMAN_PROMPT
        self.ai_prompt: Literal["\n\nAssistant:"] = AI_PROMPT

    def create_human_prompt(self, message_string: str) -> str:
        return f"{self.human_prompt}{message_string}"

    def create_ai_prompt(self, message_string: str) -> str:
        return f"{self.human_prompt}{message_string}"

    def convert_to_anthropic(self, message_dict: List[Dict[str, Any]]) -> str:
        prompt: str = ""
        for role, content in message_dict[0].items():
            if role == "user":
                prompt += self.human_prompt + content
            if role == "assistant":
                prompt += self.ai_prompt + content
        return prompt

    def convert_to_openai(
        self, message_string: str, role: Optional[str]
    ) -> List[Dict[str, str]]:
        if not role:
            role = "user"
        response = {}
        messages = message_string.split("\n\n")
        for message in messages:
            role, content = message.split(":")
            response[role.strip()] = content.strip()
        return [response]


# added token class to track tokens later on, currently serves no purpose
class AnthropicTokens:
    text = None

    def __init__(self, text):
        self.tokens = []
        self.client = AnthropicChatBot().async_claude
        self.text = text

    def sync_tokens(self, text: str):
        self.tokens = self.client.count_tokens(text)
        return f"'{text} is {self.tokens} tokens'"

    async def async_tokens(
        self,
        text: str,
    ) -> str:
        self.tokens = await self.client.count_tokens(text)
        return f"'{text} is {self.tokens} tokens'"

    async def run(self, text):
        self.sync_tokens(text)
        return self.async_tokens(text)
