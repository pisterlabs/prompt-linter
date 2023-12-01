"""Util that calls Bing Search.

In order to set this up, follow instructions at:
https://levelup.gitconnected.com/api-tutorial-how-to-use-bing-web-search-api-in-python-4165d5592a7e
"""
from typing import Dict, List
import requests
from langchain.utilities import BingSearchAPIWrapper




class BingSearchAPIMultiMarketWrapper(BingSearchAPIWrapper):
    """Wrapper for Bing Search API.

    In order to set this up, follow instructions at:
    https://levelup.gitconnected.com/api-tutorial-how-to-use-bing-web-search-api-in-python-4165d5592a7e
    """

    bing_subscription_key: str
    bing_search_url: str
    k: int = 3
    market: str = "en-US"  # 添加 market 参数，默认为 en-US


    def _bing_search_results(self, search_term: str, count: int,market: str) -> List[dict]:
        headers = {"Ocp-Apim-Subscription-Key": self.bing_subscription_key}
        params = {
            "q": search_term,
            "count": count,
            "textDecorations": True,
            "textFormat": "HTML",
            "mkt": self.market,  # 将 market 参数添加到搜索请求中
        }
        response = requests.get(
            self.bing_search_url, headers=headers, params=params  # type: ignore
        )
        response.raise_for_status()
        search_results = response.json()
        return search_results["webPages"]["value"]
    
    def run(self, query: str) -> str:
        """Run query through BingSearch and parse result."""
        snippets = []
        results = self._bing_search_results(query, count=self.k,market=self.market)
        if len(results) == 0:
            return "No good Bing Search Result was found"
        for result in results:
            snippets.append(result["snippet"])

        return " ".join(snippets)
    
    def results(self, query: str, num_results: int) -> List[Dict]:
        """Run query through BingSearch and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """
        metadata_results = []
        results = self._bing_search_results(query, count=num_results,market=self.market)
        if len(results) == 0:
            return [{"Result": "No good Bing Search Result was found"}]
        for result in results:
            metadata_result = {
                "snippet": result["snippet"],
                "title": result["name"],
                "link": result["url"],
            }
            metadata_results.append(metadata_result)

        return metadata_results
