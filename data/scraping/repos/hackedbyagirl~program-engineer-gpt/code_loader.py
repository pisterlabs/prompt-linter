#!/usr/bin/env python3

# Imports
from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..core.embedder import CodeEmbedder
from ..core.loader import CodeLoader


# Define the input schema for the tool
class CodeLoadingInput(BaseModel):
    repository: str = Field(
        description="The URL of the GitHub repository or the path of the local repository to analyze."
    )
    deeplake_username: str = Field(description="Username of deeplake account")
    dataset_name: str = Field(
        description="The name of the dataset that will be stored in DeepLake"
    )


class CodeLoaderTool(BaseTool):
    name = "CodeLoader"
    description = "Loads code from a given repository to DeepLake Vectorstore"
    args_schema: Type[BaseModel] = CodeLoadingInput

    def _run(
        self,
        repository: str,
        deeplake_username: str,
        dataset_name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        # Initialize the CodeLoader with the provided repository
        loader = CodeLoader(repository)

        # Load the repository and split the code into chunks
        code_chunks = loader.load_repository()

        # Embed the Code and Upload to DataLake VectorStore
        embed = CodeEmbedder(dataset_name, deeplake_username)
        embed.embed_and_upload(code_chunks)

        # For the sake of this example, we will just return a dummy string
        return f"Code loading for repository '{repository}' completed successfully."

    async def _arun(
        self,
        repository: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # This would be similar to the _run method, but with async calls
        # For the sake of this example, we will just return a dummy string
        return (
            f"Async code loading for repository '{repository}' completed successfully."
        )
