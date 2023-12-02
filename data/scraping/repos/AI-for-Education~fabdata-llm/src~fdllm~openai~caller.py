import os
from typing import List, Any, Union, get_args
from types import GeneratorType
import base64
from io import BytesIO
from pathlib import Path

from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from PIL import Image

from .tokenizer import tokenize_chatgpt_messages, tokenize_chatgpt_messages_v2
from ..llmtypes import (
    LLMCaller,
    LLMCallArgs,
    OpenAIModelType,
    OpenAIVisionModelType,
    AzureOpenAIModelType,
    LLMModelType,
    LLMMessage,
)


class GPTCaller(LLMCaller):
    def __init__(self, model: str = "gpt-3.5-turbo"):
        Modtype = LLMModelType.get_type(model)
        if isinstance(Modtype, tuple):
            raise ValueError(f"{model} is ambiguous type")
        if Modtype not in [OpenAIModelType, AzureOpenAIModelType]:
            raise ValueError(f"{model} is not supported")
        
        model_: LLMModelType = Modtype(Name=model)
        
        if Modtype in [OpenAIModelType]:
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)
        elif Modtype in [AzureOpenAIModelType]:
            client = AzureOpenAI(azure_deployment=model, **model_.Client_Args)
            aclient = AsyncAzureOpenAI(azure_deployment=model, **model_.Client_Args)
        super().__init__(
            Model=model_,
            Func=client.chat.completions.create,
            AFunc=aclient.chat.completions.create,
            Args=LLMCallArgs(
                Model="model",
                Messages="messages",
                Max_Tokens="max_tokens",
            ),
            Defaults={},
            Token_Window=model_.Token_Window,
            Token_Limit_Completion=model_.Token_Limit_Completion,
        )

    def format_message(self, message: LLMMessage):
        return {"role": message.Role, "content": message.Message}

    def format_messagelist(self, messagelist: List[LLMMessage]):
        return [self.format_message(message) for message in messagelist]

    def format_output(self, output: Any):
        if isinstance(output, GeneratorType):
            return output
        else:
            return LLMMessage(
                Role="assistant", Message=output.choices[0].message.content
            )

    def tokenize(self, messagelist: List[LLMMessage]):
        return tokenize_chatgpt_messages(self.format_messagelist(messagelist))[0]


class GPTVisionCaller(LLMCaller):
    def __init__(
        self,
        model: str = "gpt-4-vision-preview"
    ):
        Modtype = LLMModelType.get_type(model)
        if isinstance(Modtype, tuple):
            raise ValueError(f"{model} is ambiguous type")
        if Modtype not in [OpenAIVisionModelType]:
            raise ValueError(f"{model} is not supported")
        
        model_: LLMModelType = Modtype(Name=model)
        
        if Modtype in [OpenAIVisionModelType]:
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)
            
        super().__init__(
            Model=model_,
            Func=client.chat.completions.create,
            AFunc=aclient.chat.completions.create,
            Args=LLMCallArgs(
                Model="model",
                Messages="messages",
                Max_Tokens="max_tokens",
            ),
            Defaults={},
            Token_Window=model_.Token_Window,
            Token_Limit_Completion=model_.Token_Limit_Completion,
        )

    def format_message(self, message: LLMMessage):
        content = [{"type": "text", "text": message.Message}]
        if message.Images is not None:
            content += [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            im.Url
                            if im.Url is not None
                            else f"data:image/png;base64,{im.encode()}"
                        ),
                        "detail": im.Detail,
                    },
                }
                for im in message.Images
            ]
        return {"role": message.Role, "content": content}

    def format_messagelist(self, messagelist: List[LLMMessage]):
        return [self.format_message(message) for message in messagelist]

    def format_output(self, output: Any):
        if isinstance(output, GeneratorType):
            return output
        else:
            return LLMMessage(
                Role="assistant", Message=output.choices[0].message.content
            )

    def tokenize(self, messagelist: List[LLMMessage]):
        texttokens = tokenize_chatgpt_messages_v2(self.format_messagelist(messagelist))
        imgtokens = 0
        for msg in messagelist:
            if msg.Images is not None:
                for img in msg.Images:
                    ntok = img.tokenize()
                    imgtokens += ntok
        return [None] * (texttokens + imgtokens)
