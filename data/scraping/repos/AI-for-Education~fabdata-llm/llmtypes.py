from __future__ import annotations
from typing import List, Literal, Callable, Awaitable, Any, Optional, Dict, Union
import datetime
from abc import ABC, abstractmethod
import os
from dataclasses import field
from functools import wraps
from pathlib import Path
from io import BytesIO
import base64

import numpy as np
from openai import RateLimitError as RateLimitErrorOpenAI
from anthropic import RateLimitError as RateLimitErrorAnthropic
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, BaseModel, Field
from PIL import Image

from .decorators import delayedretry
from .openai.tokenizer import tokenize_chatgpt_messages
from .constants import LLM_DEFAULT_MAX_TOKENS, LLM_DEFAULT_MAX_RETRIES
from .sysutils import load_models


class LLMModelType(BaseModel):
    Name: Optional[str]
    Token_Window: int
    Token_Limit_Completion: Optional[int] = None
    Client_Args: dict = Field(default_factory=dict)

    def __init__(self, Name, model_type=None):
        if model_type is None:
            super().__init__(Name=Name, Token_Window=0)
            return
        basemodels = load_models()
        if model_type not in basemodels:
            raise NotImplementedError(f"{model_type} is not a valid model type")
        typemodels = basemodels[model_type]
        if Name not in typemodels:
            raise NotImplementedError(f"{Name} is not a valid {model_type} model")
        super().__init__(Name=Name, **typemodels[Name])

    @classmethod
    def get_type(cls, name) -> Union[tuple[LLMModelType], LLMModelType, None]:
        basemodels = load_models()
        usetype = []
        for model_type, models in basemodels.items():
            if name in models:
                usetype.append(model_type)
        modtypelist = []
        for modtype in usetype:
            if modtype == "OpenAI":
                modtypelist.append(OpenAIModelType)
            elif modtype == "OpenAIVision":
                modtypelist.append(OpenAIVisionModelType)
            elif modtype == "AzureOpenAI":
                modtypelist.append(AzureOpenAIModelType)
            elif modtype == "Anthropic":
                modtypelist.append(AnthropicModelType)
        if len(modtypelist) > 1:
            return tuple(modtypelist)
        elif len(modtypelist) == 1:
            return modtypelist[0]
        

class OpenAIModelType(LLMModelType):
    def __init__(self, Name):
        super().__init__(Name, "OpenAI")

class OpenAIVisionModelType(LLMModelType):
    def __init__(self, Name):
        super().__init__(Name, "OpenAIVision")

class AzureOpenAIModelType(LLMModelType):
    def __init__(self, Name):
        super().__init__(Name, "AzureOpenAI")

class AnthropicModelType(LLMModelType):
    def __init__(self, Name):
        super().__init__(Name, "Anthropic")


class LLMMessage(BaseModel):
    Role: Literal["user", "assistant", "system", "error"]
    Message: str
    Images: Optional[List[LLMImage]] = None
    TokensUsed: int = 0
    DateUTC: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    def __eq__(self, __value: object) -> bool:
        # exclude timestamp from equality test
        if isinstance(__value, self.__class__):
            return (
                self.Role == __value.Role
                and self.Message == __value.Message
                and self.Images == __value.Images
                and self.TokensUsed == __value.TokensUsed
            )
        else:
            return super().__eq__(__value)

    class Config:
        arbitrary_types_allowed = True


class LLMImage(BaseModel):
    Url: Optional[str] = None
    Img: Optional[Union[Image.Image, Path]] = None
    Detail: Literal["low", "high"] = "low"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._format_image()

    @classmethod
    def list_from_images(
        cls, images: Optional[List[Image.Image]], detail: Literal["low", "high"] = "low"
    ):
        if images is None:
            return
        else:
            return [cls(Img=img, Detail=detail) for img in images]

    def encode(self):
        if self.Img is None:
            return
        bts = BytesIO()
        self.Img.save(bts, format="png")
        bts.seek(0)
        return base64.b64encode(bts.read()).decode("utf-8")

    def tokenize(self):
        if self.Detail == "low":
            return 85
        else:
            if self.Img is None:
                # if image is url we use worst case scenario
                # for width and height
                width, height = 2048, 768
            else:
                width, height = self.Img.size
            ngridw = int(np.ceil(width / 512))
            ngridh = int(np.ceil(height / 512))
            ntiles = ngridw * ngridh
            return ntiles * 170 + 85

    def _format_image(self):
        im = self.Img
        if im is None:
            return
        detail = self.Detail
        width, height = im.size
        if detail == "low":
            im = im.resize((512, 512), Image.BILINEAR)
        else:
            maxdim = max(width, height)
            if maxdim > 2048:
                width = width * (2048 / maxdim)
                height = height * (2048 / maxdim)
            shortestdim = min(width, height)
            scale = min(768 / shortestdim, 1)
            finalwidth = int(width * scale)
            finalheight = int(height * scale)
            im = im.resize((finalwidth, finalheight), Image.BILINEAR)
        self.Img = im

    class Config:
        arbitrary_types_allowed = True


@dataclass(config=ConfigDict(validate_assignment=True))
class LLMCallArgs:
    Messages: str
    Model: str
    Max_Tokens: str


class LLMCaller(ABC, BaseModel):
    Model: LLMModelType
    Func: Callable[..., Any]
    AFunc: Callable[..., Awaitable[Any]]
    Token_Window: int
    Token_Limit_Completion: Optional[int] = None
    Defaults: Dict = Field(default_factory=dict)
    Args: Optional[LLMCallArgs] = None

    @abstractmethod
    def format_message(self, message: LLMMessage):
        pass

    @abstractmethod
    def format_messagelist(self, messagelist: List[LLMMessage]):
        pass

    @abstractmethod
    def format_output(self, output: Any) -> LLMMessage:
        pass

    @abstractmethod
    def tokenize(self, messagelist: List[LLMMessage]) -> List[int]:
        pass

    def sanitize_messagelist(
        self, messagelist: List[LLMMessage], min_new_token_window: int
    ) -> List[LLMMessage]:
        out = messagelist
        while (
            self.Token_Window - len(self.tokenize(messagelist)) < min_new_token_window
        ):
            out = out[1:]
        return out

    def call(
        self,
        messages: List[LLMMessage] | LLMMessage,
        max_tokens: Optional[int] = LLM_DEFAULT_MAX_TOKENS,
        **kwargs,
    ):
        kwargs = self._proc_call_args(messages, max_tokens, **kwargs)
        return self.format_output(self._call(**kwargs))

    async def acall(
        self,
        messages: List[LLMMessage] | LLMMessage,
        max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
        **kwargs,
    ):
        kwargs = self._proc_call_args(messages, max_tokens, **kwargs)
        return self.format_output(await self._acall(**kwargs))

    def _proc_call_args(self, messages, max_tokens, **kwargs):
        if isinstance(messages, LLMMessage):
            messages = [messages]
        if max_tokens is None:
            max_tokens = self.Token_Window - (len(self.tokenize(messages)) + 64)
        if self.Token_Limit_Completion is not None:
            max_tokens = min(max_tokens, self.Token_Limit_Completion)
        if self.Args is not None:
            kwargs[self.Args.Model] = self.Model.Name
            kwargs[self.Args.Max_Tokens] = max_tokens
            kwargs[self.Args.Messages] = self.format_messagelist(messages)
        return {**self.Defaults, **kwargs}

    @delayedretry(
        rethrow_final_error=True,
        max_attempts=LLM_DEFAULT_MAX_RETRIES,
        include_errors=[RateLimitErrorOpenAI, RateLimitErrorAnthropic],
    )
    def _call(self, *args, **kwargs):
        return self.Func(*args, **kwargs)

    @delayedretry(
        rethrow_final_error=True,
        max_attempts=LLM_DEFAULT_MAX_RETRIES,
        include_errors=[RateLimitErrorOpenAI, RateLimitErrorAnthropic],
    )
    async def _acall(self, *args, **kwargs):
        return await self.AFunc(*args, **kwargs)


class LiteralCaller(LLMCaller):
    def __init__(self, text: str):
        super().__init__(
            Model=LLMModelType(Name=None),
            Func=lambda: text,
            AFunc=self._literalafunc(text),
            Token_Window=0,
        )

    @staticmethod
    def _literalafunc(text):
        async def afunc():
            return text

        return afunc

    def format_message(self, message: LLMMessage):
        return super().format_message(message)

    def format_messagelist(self, messagelist: List[LLMMessage]):
        return super().format_messagelist(messagelist)

    def format_output(self, output: Any) -> LLMMessage:
        return LLMMessage(Role="assistant", Message=output)

    def tokenize(self, messagelist: List[LLMMessage]) -> List[int]:
        return super().tokenize(messagelist)


LLMMessage.update_forward_refs()
