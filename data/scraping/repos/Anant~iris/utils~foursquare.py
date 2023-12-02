"""Chain that calls FoursquaerAPI.

"""
import requests
import json
import os
import sys
from typing import Any, Dict, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env

def search_places(query: str, api_key: str, limit: int) -> dict:
        """Search for places on Foursquare given a query and API key."""
        url = "https://api.foursquare.com/v3/places/search"
        headers = {
            "Authorization": api_key,
            "accept": "application/json"
        }
        params = {
            "query": query,
            "limit": limit
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

def search_places_near(query: str, near: str, api_key: str, limit: int) -> dict:
        """Search for places on Foursquare given a query and where and API key."""
        url = "https://api.foursquare.com/v3/places/search"
        headers = {
            "Authorization": api_key,
            "accept": "application/json"
        }
        params = {
            "query": query,
            "near" : near,
            "limit": limit
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout


class FoursquareAPIWrapper(BaseModel):
    """Wrapper around Foursquare API.

    To use, you should have the environment variable ``FOURSQUARE_API_KEY`` set with your API key, or pass
    `foursquare_api_key` as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain import FoursquareAPIWrapper
            foursquareapi = FoursquareAPIWrapper()
    """

    search_engine: Any  #: :meta private:

    foursquare_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid


    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        foursquare_api_key = get_from_dict_or_env(
            values, "foursquare_api_key", "FOURSQUARE_API_KEY"
        )
        values["foursquare_api_key"] = foursquare_api_key

        try:
            values["foursquare_engine"] = search_places(query="coffee", api_key=values["foursquare_api_key"], limit=1)
            print("Search successful!")
            # Do something with the result, such as parse the JSON or display it to the user
        except requests.exceptions.HTTPError as e:
            print(f"Search failed: {e}")
            ValueError(f"Got error from Foursquare: {e}")
            # Handle the error, such as displaying an error message to the user or logging the error    
        return values
    
    def run(self, query: str) -> str:
        """Run query through FoursquareAPI and parse result."""
        api_key = self.foursquare_api_key  # str | Foursquare API Key.
        q = query  # str | Search query term or phrase.
        # int | The maximum number of records to return. 
        limit = 5

        with HiddenPrints():
            try:
                # Search Endpoint
                api_response = search_places_near(query=q, near="Washington, DC",api_key=api_key, limit=limit)  
            except requests.exceptions.HTTPError as e:
                raise ValueError(f"Got error from FoursquareAPI: {e}")

        #name = api_response.data["results"][0].name
        result_json = json.dumps(api_response, separators=(",", ":"))
        return f"""Output: {result_json}"""

    
    def near(self, query: str) -> str:
        """Run query through FoursquareAPI and parse result."""
        api_key = self.foursquare_api_key  # str | Foursquare API Key.
        q,n = query.split("|")  # str | Search query term or phrase.
        
        # int | The maximum number of records to return. 
        limit = 5

        with HiddenPrints():
            try:
                # Search Endpoint
                api_response = search_places_near(query=q, near=n,api_key=api_key, limit=limit)  
            except requests.exceptions.HTTPError as e:
                raise ValueError(f"Got error from FoursquareAPI: {e}")

        #name = api_response.data["results"][0].name
        result_json = json.dumps(api_response, separators=(",", ":"))
        return f"""Output: {result_json}"""
