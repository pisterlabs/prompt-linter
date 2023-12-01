"""Functional call"""
import os
from typing import Optional
from rich import print

import langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.tools import format_tool_to_openai_function, BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

langchain.debug = True
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
os.environ["OPENAI_API_TYPE"] = "open_ai"
os.environ["OPENAI_API_VERSION"] = ""


class CodeDebugTool(BaseTool):
    name = "code_debug"
    description = "Use this tool when you want to debug the code."

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return "This is a code debug tool."

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("code_debug does not support async")


class CodeReviewTool(BaseTool):
    name = "code_review"
    description = "Use this tool when you want to review the code."

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return "This is a code reiview tool."

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("code_review does not support async")


class CodeQualityCheckerTool(BaseTool):
    name = "code_quality_checker"
    description = "Use this tool when you want to check the quality of the code."

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return "This is a code quality check tool."

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("code_quality_checker does not support async")


tools = [
    CodeQualityCheckerTool(),
    CodeDebugTool(),
    CodeReviewTool(),
]
functions = [format_tool_to_openai_function(t) for t in tools]
model = ChatOpenAI(model="gpt-3.5-turbo-0613", verbose=True)
content = """
import abc

from prompt import log


class BaseBot(metaclass=abc.ABCMeta):
    def query(self, question: str) -> dict:
        return {}

    def get_response(self, completion) -> str:
        log.logger.info({"msg": "ChatGPT response completion", "completion": str(completion)})
        return completion.choices[0].message.content.replace("\n", " ")

"""

import openai

print(openai.api_key)
print(openai.api_base)
print(openai.api_type)
print(openai.api_version)
message = model.predict_messages([HumanMessage(content=content)], functions=functions)
print(f"{message = }")
