from enum import Enum
from typing import Tuple, List, Dict

import base64
import discord
import requests
import tiktoken
from openai import AsyncOpenAI, RateLimitError, APIStatusError


class BotGPTException(Exception):
    pass


class GPTModel(Enum):
    """
    Enum class for different GPT models with their corresponding name, token limit, and emoji name
    """
    CHAT_GPT = ('gpt-3.5-turbo', 4096, True)
    CHAT_GPT_16K = ('gpt-3.5-turbo-1106', 16385, True)
    GPT_4_VISION = ('gpt-4-vision-preview', 128000, True)
    GPT_4_TURBO = ('gpt-4-1106-preview', 128000, True)

    def __init__(self, version: str, token_limit: int, available: bool):
        self.version = version
        self.token_limit = token_limit
        self.available = available

    @classmethod
    def from_version(cls, version: str):
        """
        Retrieve the GPTModel enum member based on the provided version string.

        Parameters:
            version: The version string to search for in the enum.

        Returns:
            GPTModel: The GPTModel enum member corresponding to the given version.

        Raises:
            ValueError: If the provided version is not found in the enum.
        """
        for member in cls:
            if member.version == version:
                return member
        raise ValueError(f'Non-existing "{version}" version. Available versions are {[m.version for m in cls]}')


class GPT:
    """A class to encapsulate OpenAI GPT related functionalities."""

    DEFAULT_MODEL = GPTModel.GPT_4_TURBO
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_TOP_P = 1.0

    def __init__(self, api_key):
        self.client = AsyncOpenAI(api_key=api_key)

    async def communicate(self, thread: discord.Thread) -> str:
        """
        Collects GPT payload for the given thread and sends it to OpenAI chat completion API.

        Args:
            thread (discord.Thread): the thread to fetch messages from.

        Returns:
            gpt_response (str): the generated gpt response.
        """
        model, temperature, top_p, messages = await self._collect_gpt_payload(thread)
        gpt_response = await self._send_payload(model, temperature, top_p, messages)
        return gpt_response

    async def _collect_gpt_payload(self, thread: discord.Thread) -> Tuple[GPTModel, float, float, List[Dict]]:
        """
        Fetches origin data from the thread and forms message history.

        Args:
            thread (discord.Thread): the thread to fetch messages from.

        Returns:
            model (GPTModel): the OpenAI model extracted from the thread.
            temperature (float): temperature setting for the model.
            top_p (float): top_p setting for the model.
            messages (List[dict]): the message history in the format suitable for GPT.
        """
        starter_message, model_message, temperature_message, top_p_message = [m async for m in
                                                                              thread.history(limit=4,
                                                                                             oldest_first=True)]
        # starter_message is not cached
        if not starter_message:
            starter_message = await thread.parent.fetch_message(starter_message.id)

        # Extract configurations
        from discord_util import DiscordUtil
        model = GPTModel.from_version(DiscordUtil.extract_set_value(model_message))
        temperature = float(DiscordUtil.extract_set_value(temperature_message))
        top_p = float(DiscordUtil.extract_set_value(top_p_message))

        messages = []
        tokens = 0

        # Add GPT's system message
        system_message_content = [{
            'type': 'text',
            'text': starter_message.system_content
        }]
        system_message = {'role': 'system', 'content': starter_message.system_content}
        tokens += await self._calculate_tokens(system_message_content, model)
        messages.append(system_message)
        reached_token_limit = False
        # Fetches history in reverse order
        async for msg in thread.history():
            # Skip configuration messages
            if msg in [starter_message, model_message, temperature_message, top_p_message]:
                continue
            entries = []
            # Handle message attachments
            for attachment in msg.attachments:
                content = await self._handle_attachment(attachment)
                entry = {
                    'role': 'assistant' if msg.author.bot else 'user',
                    'content': content
                }
                entries.append(entry)
                tokens += await self._calculate_tokens(content, model)
                if tokens > model.token_limit:
                    reached_token_limit = True
            # Handle actual message content
            if msg.content:
                content = [{
                    'type': 'text',
                    'text': msg.content
                }]
                entry = {
                    'role': 'assistant' if msg.author.bot else 'user',
                    'content': content
                }
                entries.append(entry)
                tokens += await self._calculate_tokens(content, model)
                if tokens > model.token_limit:
                    reached_token_limit = True
            if reached_token_limit:
                break
            # Insert at the beginning
            messages[1:1] = entries

        return model, temperature, top_p, messages

    async def _send_payload(self, model: GPTModel,
                            temperature: float,
                            top_p: float,
                            messages: List[Dict]) -> str:
        """
        Get response from OpenAI GPT API.

        Args:
            messages (List[dict]): the message history in the format suitable for GPT.
            temperature (float): temperature setting for the model.
            top_p (float): top_p setting for the model.
            model (GPTModel): the OpenAI model extracted from the thread.

        Returns:
            assistant_response (str): the response content from the API.
        """
        response = await self.client.chat.completions.create(
            model=model.version,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=4096 if model == GPTModel.GPT_4_VISION else None
        )
        assistant_response = response.choices[0].message.content
        return assistant_response

    async def _calculate_tokens(self, content: list[dict[str, str]], model: GPTModel) -> int:
        """
        Calculates the number of tokens required to process a message.

        Args:
            msg (str): the message to process.
            model (GPTModel): the OpenAI model to use.

        Returns:
            num_tokens (int): the number of tokens required to process the message.
        """
        try:
            encoding = tiktoken.encoding_for_model(model.version)
            tokens = 0
            for entry in content:
                if entry['type'] == 'text':
                    tokens += len(encoding.encode(entry['text']))
                elif entry['type'] == 'image_url':
                    """TODO: Calculate image tokens as per 
                    https://platform.openai.com/docs/guides/vision/calculating-costs"""
                    pass
            return tokens
        except KeyError:
            raise NotImplementedError(
                f'_calculate_tokens() is not presently implemented for model {model.version}'
            )

    async def _handle_attachment(self, attachment: discord.Attachment) -> list[dict[str, str]]:
        content_type = attachment.content_type
        if 'text/plain' in content_type:
            response = requests.get(attachment.url)
            if response.status_code == 200:
                return [{
                    'type': 'text',
                    'text': response.text
                }]
            else:
                raise BotGPTException(f'Failed to download attachment: {response.status_code}')
        elif content_type in ('image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'):
            response = requests.get(attachment.url)
            if response.status_code == 200:
                # Convert the image content to base64
                base64_image = base64.b64encode(response.content).decode('utf-8')
                return [{
                    'type': 'image_url',
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }]
            else:
                raise BotGPTException(f'Failed to download attachment: {response.status_code}')
        else:
            raise BotGPTException(f'Unsupported attachment type: {content_type}')
