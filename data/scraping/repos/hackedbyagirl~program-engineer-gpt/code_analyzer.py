#!/usr/bin/env python3

# Imports

from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..core.analyzer import AnalyzeCode


# Define the input schema for the tool
class CodeAnalyzingInput(BaseModel):
    username: str = Field(description="The username of the DeepLake account.")
    dataset_name: str = Field(description="The name of the dataset in DeepLake.")
    model_name: str = Field(description="The name of the model to use for analysis.")
    question: str = Field(description="The question to ask about the code.")


class CodeAnalyzerTool(BaseTool):
    name = "CodeAnalyzer"
    description = "Analyzes code from a given dataset in DeepLake"
    args_schema: Type[BaseModel] = CodeAnalyzingInput

    def _run(
        self,
        username: str,
        dataset_name: str,
        model_name: str,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        # Initialize the AnalyzeCode with the provided parameters
        analyzer = AnalyzeCode(username, dataset_name, model_name)

        # Ask the question and get the answer
        answer = analyzer.ask_question(question)

        # Return the answer
        return answer

    async def _arun(
        self,
        username: str,
        dataset_name: str,
        model_name: str,
        question: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # This would be similar to the _run method, but with async calls
        # For the sake of this example, we will just return a dummy string
        return (
            f"Async code analyzing for dataset '{dataset_name}' completed successfully."
        )
