import json
import os
from typing import Type

import requests
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.base import BaseTool

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


class GoogleSearchInput(BaseModel):
    query: str = Field(..., description="Query for search.")
    num_results: int = Field(..., description="Number of results to be returned")


class GoogleSearchTool(BaseTool):
    name: str = "google_search"
    description: str = (
        "A wrapper around Google Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. Output is a JSON array of the query results"
    )
    args_schema: Type[BaseModel] = GoogleSearchInput

    def _run(
        self,
        query: str,
        num_results: int,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CSE_ID,
                "q": query,
                "num": num_results,
            },
        )
        result = response.json()
        ret = []
        for item in result["items"]:
            print(item)
            ret.append(
                {
                    "title": item["title"],
                    "link": item["link"],
                    "snippet": item.get("snippet"),
                }
            )
        return json.dumps(ret)
