# -*- coding: utf-8 -*-
import json
from typing import List
import openai
import anthropic
from pydantic import BaseModel
from abc import ABC, abstractmethod
import sys
import os
from typing import Callable, Optional, Sequence, List
import tiktoken
import asyncio
import transformers
import logging
from typing import Self
sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
from oneapi import OneAPITool, register_client, AbstractClient, AbstractConfig
class MockClient(AbstractClient):

        def __init__(self, method: AbstractConfig) -> None:
                super().__init__(method)
                self.method = method
                self.client = None
                self.aclient = None

        @classmethod
        def from_config(cls, config: dict = None, config_file: str = "") -> Self:
                return cls(AbstractConfig(**config))

        def format_prompt(self, prompt: str | list[str] | list[dict], system: str = ""):
                pass
                
        def chat(self, prompt: str | list[str] | list[dict], system: str = "", **kwargs):
                return prompt
        
        def achat(self, prompt: str | list[str] | list[dict], system: str = "", max_tokens: int = 1024, **kwargs):
                pass

        
        def count_tokens(self, texts: List[str], model: str = "") -> int:
                pass
register_client("mock", MockClient)

tool = OneAPITool.from_config(api_key="", api_type="claude", api_base="https://api.anthropic.com", api_version="mock")
print(tool.chat("hello AI"))



