import asyncio
import json
from typing import Any, Dict, List, Literal, Optional, Union
import aiohttp
from datetime import datetime, timezone
from gptmod.object_core import ApiCore
from gptmod.api import GptmodAPI
from gptmod.util import num_tokens_from_messages
import openai


class Image(ApiCore):
    endpoint = "v1/images/generations"
    method = "POST"
    api_slots = []

    def __init__(
        self,
        prompt: str,
        n: int = 1,
        user: Optional[str] = None,
        size: Literal["256x256", "512x512", "1024x1024"] = "256x256",
    ):
        self.prompt = prompt
        self.n = n
        self.size = size
        self.user = user

    async def calloai(self):
        """return a completion through openai instead."""
        dictme = self.to_dict()
        result = await openai.Image.acreate(**dictme)
        return result

    def to_dict(self):
        data = super().to_dict()
        return data


class ImageVariate(ApiCore):
    endpoint = "images/variations"
    method = "POST"
    api_slots = []

    def __init__(
        self,
        image: str,
        n: int = 1,
        user: Optional[str] = None,
        size: Literal["256x256", "512x512", "1024x1024"] = "256x256",
    ):
        self.image = image
        self.n = n
        self.size = size
        self.user = user

    async def calloai(self):
        """return a completion through openai instead."""
        dictme = self.to_dict()
        result = await openai.Image.acreate_variation(**dictme)
        return result

    def to_dict(self):
        data = super().to_dict()
        return data


class Edit(ApiCore):
    endpoint = "edits"
    method = "POST"
    api_slots = []

    def __init__(
        self,
        model: str,
        input: str,
        instruction: str,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        self.model: str = (model,)
        self.input = input
        self.instruction = instruction
        self.n = n
        self.temperature = temperature
        self.top_p = top_p

    def to_dict(self):
        dictv = {
            "model": "text-davinci-edit-001",
            "input": self.input,
            "instruciton": self.instruction,
        }
        return dictv
