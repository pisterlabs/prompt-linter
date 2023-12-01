"""Chain that calls GiphyAPi.

"""
import json
import os
import sys
import traceback
from typing import Any, Dict, Optional

import giphy_client
from giphy_client.rest import ApiException
from pydantic import BaseModel, Extra, root_validator

from langchain.agents import Tool
from langchain.utils import get_from_dict_or_env

from ..utils.helper import parse_json_string, random_number


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


class GiphyAPIWrapper(BaseModel):
    """Wrapper around Giphy API.

    To use, you should have the environment variable ``GIPHY_API_KEY`` set with your API key, or pass
    `giphyapi_api_key` as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain import SerpAPIWrapper
            serpapi = SerpAPIWrapper()
    """

    search_engine: Any  #: :meta private:

    giphy_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        giphy_api_key = get_from_dict_or_env(values, "giphy_api_key", "GIPHY_API_KEY")
        values["giphy_api_key"] = giphy_api_key
        try:
            values["giphy_engine"] = giphy_client.DefaultApi()
        except ImportError:
            raise ValueError(
                "Could not import giphy_client python package. "
                "Please it install it with `pip install giphy_client`."
            )
        return values

    def run(self, query: str) -> str:
        try:
            input = parse_json_string(query)

            input["offset"] = 0 if not input["random"] else random_number(0, 100)

            api_type = input["api_type"]

            # API Endpoints: https://developers.giphy.com/docs/api/endpoint/

            """Run query through GiphyAPI and parse result."""
            api_key = self.giphy_api_key  # str | Giphy API Key.

            with HiddenPrints():
                try:
                    if api_type in ["trending"]:
                        api_response = self.giphy_engine.gifs_trending_get(
                            api_key,
                            limit=input["limit"],
                            offset=input["offset"],
                        )
                    elif api_type in ["translate"]:
                        api_response = self.giphy_engine.gifs_translate_get(
                            api_key,
                            s=input["q"],
                        )
                        api_response.data = [api_response.data]
                    else:
                        api_response = self.giphy_engine.gifs_search_get(
                            api_key,
                            q=input["q"],
                            limit=input["limit"],
                            offset=input["offset"],
                        )
                except ApiException as e:
                    raise ValueError(f"Got error from GiphyAPI: {e}")

            return {
                "success": True,
                "input": input,
                "giphy": list(map(lambda x: x.to_dict(), api_response.data)),
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "success": False,
                "error": e.args,
            }


giphy = Tool(
    name="giphy",
    func=GiphyAPIWrapper().run,
    description="useful for when you need to find gif or clips. The input of this tool in json format only with q key as description of the image; limit key as number of result in integer; api_type key as one of these values: search, trending; random key as user requests random results or not in boolean format.",
    return_direct=True,
)
