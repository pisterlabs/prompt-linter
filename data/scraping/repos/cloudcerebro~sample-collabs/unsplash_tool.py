"""Tool for the Unsplash image search API."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pydantic import Field
from langchain.tools.base import BaseTool
from unsplash_search import UnsplashSearchAPIWrapper


class UnsplashImageSearchTool(BaseTool):
    """Tool that adds the capability to query the Unsplash image search API."""

    name = "Unsplash Image Search"
    description = (
        "A wrapper around Unsplash Image Search. "
        "Useful for when you need to find images related to a specific topic. "
        "Input should be an image search query."
    )
    api_wrapper: UnsplashSearchAPIWrapper

    def __init__(self, api_key: str):
        super().__init__(name="Unsplash Image Search", 
                         description="A wrapper around Unsplash Image Search. Useful for when you need to find images related to a specific topic. Input should be an image search query.",
                         api_wrapper=UnsplashSearchAPIWrapper(api_key=api_key))

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("UnsplashImageSearch does not support async")
