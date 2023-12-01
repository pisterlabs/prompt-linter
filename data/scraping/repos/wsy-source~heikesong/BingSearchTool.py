import os
from typing import Any
from langchain.tools import BaseTool
import requests


class BingSearchTool(BaseTool):
    name = "SearchTool"
    description = "Useful tools provide users with links to real-time data"

    def __init__(self):
        self.subscription_key = os.getenv("SEARCH_SUBSCRIPTION_KEY", "8f324c7ee4db4faab7390b9d29ccac4b")
        self.endpoint = os.getenv("SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")

    def _run(self, query: str) -> Any:
        mkt = 'en-US'
        params = {'q': query, 'mkt': mkt}
        headers = {'Ocp-Apim-Subscription-Key': self.subscription_key}

        # Call the API
        try:
            name = ""
            url = ""
            response = requests.get(self.endpoint, headers=headers, params=params)
            response.raise_for_status()
            content = response.json()
            for value in content["webPages"]["value"]:
                name = value["name"]
                url = value["url"]
        except Exception as ex:
            raise ex
        return "search name:" + name + "url:" + url


    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        pass
