import re
from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain.tools import BaseTool, DuckDuckGoSearchResults
from pydantic import BaseModel


class SearchInternetTool(BaseTool):
    name = "search_internet_tool"
    description = "Search the internet for the up-to-date information."

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously."""
        search = DuckDuckGoSearchResults()
        results = search.run(query)

        pattern = r"\[snippet: (.*?), title: (.*?), link: (.*?)\]"
        matches = re.findall(pattern, results)

        docs = [
            {"snippet": match[0], "title": match[1], "link": match[2]}
            for match in matches
        ]

        docs_list = []

        for doc in docs:
            docs_list.append(
                {
                    "content": doc["snippet"],
                    "source": "[{}]({})".format(doc["title"], doc["link"]),
                }
            )

        return docs_list

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        search = DuckDuckGoSearchResults()
        results = search.run(query)

        pattern = r"\[snippet: (.*?), title: (.*?), link: (.*?)\]"
        matches = re.findall(pattern, results)

        docs = [
            {"snippet": match[0], "title": match[1], "link": match[2]}
            for match in matches
        ]

        docs_list = []

        for doc in docs:
            docs_list.append(
                {
                    "content": doc["snippet"],
                    "source": "[{}]({})".format(doc["title"], doc["link"]),
                }
            )

        return docs_list
