#!/usr/bin/env python3

# imports
from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from langchain.vectorstores import DeepLake
from openai import OpenAI
from pydantic import BaseModel, Field


# Define the input schema for the tool
class CodeWritingInput(BaseModel):
    task: str = Field(description="The code task to generate code for")
    username: str = Field(description="The username of the DeepLake account.")
    dataset_path: str = Field(description="The path to the dataset in DeepLake")


class CodeWriterTool(BaseTool):
    name = "CodeWriter"
    description = "Generates code based on a provided task"
    args_schema: Type[BaseModel] = CodeWritingInput

    def _run(
        self,
        task: str,
        username: str,
        dataset_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        # Initialize the OpenAI API with your API key
        openai = OpenAI(api_key="your-api-key")

        # Retrieve the embeddings from DeepLake
        db = DeepLake(dataset_path)
        embeddings = db.get_all_embeddings()

        # Convert the embeddings to a string representation
        embeddings_str = str(embeddings)

        # Use the chat models to generate code, with the embeddings as context
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "system", "content": embeddings_str},
                {"role": "user", "content": task},
            ],
        )

        # Return the generated code
        return response["choices"][0]["message"]["content"]

    async def _arun(
        self,
        task: str,
        dataset_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # This would be similar to the _run method, but with async calls
        # For the sake of this example, we will just return a dummy string
        return f"Async code generation for task '{task}' completed successfully."
