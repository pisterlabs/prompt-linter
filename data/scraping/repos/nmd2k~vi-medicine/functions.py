import sys
sys.path.append(["../"])

from pathlib import Path
from typing import Any, List, Dict

from langchain.schema import HumanMessage, SystemMessage

from utils.prompts import *
from utils.chatmodel import ChatModel
from app.exception.custom_exception import CustomException

class Functions:
    "Simple generate function and embedding functions"
    def __init__(self,
        top_p: float = 1,
        max_tokens: int = 512,
        temperature: float = 0,
        n_retry: int = 2,
        request_timeout: int = 30, **kwargs) -> None:
        
        self.chatmodel = ChatModel(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n_retry=n_retry,
            request_timeout=request_timeout,
            **kwargs
        )

    async def generate(self, message, prompt=None):
        """
        Chat model generate function
        Args:
            - message (str): human query/message
            - prompt (str): optional system message
        Return:
            - str: Generated output
        """
        try:
            messages = []
            if prompt:
                messages.append(SystemMessage(content=prompt))
            messages.append(HumanMessage(content=message))
            generate = self.chatmodel.generate(messages)
        
            return generate
        except Exception as exc:
            raise CustomException(exc)
        
    async def embed(self, message):
        """
        Embedding string input
        Args:
            - message (str): message to embed
        Return:
            - List: List of embedding output
        """
        try:
            assert type(message) == str
            embed = self.chatmodel.embed(message=message)
            print(len(embed))
            return embed
        except Exception as exc:
            raise CustomException(exc)
