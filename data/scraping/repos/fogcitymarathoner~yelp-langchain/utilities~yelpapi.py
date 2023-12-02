"""Chain that calls YelpAPI.

Heavily borrowed from https://github.com/ofirpress/self-ask
"""
import os
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain.utils import get_from_dict_or_env
from pydantic import BaseModel, Extra, Field, root_validator
from yelpapi import YelpAPI

load_dotenv()

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


class YelpAPIWrapper(BaseModel):
    """Wrapper around YelpAPI.
    To use, you should have the ``yelpapi`` python package installed
    Requires the environment variables ``YELP_API_KEY`` set.

    Example:
        .. code-block:: python

            from utilities.yelpapi import YelpAPIWrapper
            yelpapi = YelpAPIWrapper()
    """

    params: dict = Field(
        default={
            "name": "Splash Cafe",
            "address1": "197 Pomeroy Ave",
            "city": "Pismo Beach",
            "state": "CA",
            "country": "US",
        }
    )
    yelpapi_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        yelpapi_api_key = get_from_dict_or_env(
            values, "yelpapi_api_key", "YELP_API_KEY"
        )
        values["yelpapi_api_key"] = yelpapi_api_key
        try:
            from yelpapi import YelpAPI
        except ImportError:
            raise ValueError(
                "Could not import yelpapi python package. "
                "Please install it with `pip install yelpapi`."
            )
        return values
    def run(self, **kwargs: Any) -> str:
        """Run query through YelpAPI and parse result."""
        return self._process_response(self.results(kwargs['name'], kwargs['address1'], kwargs['city'], kwargs['state'], kwargs['country']))

    def results(self, name: str, address1: str, city: str, state:str, country='US') -> dict:
        """Run query through YelpAPI and return the raw result."""
        params = self.get_params(name, address1, city, state, country)
        with HiddenPrints():
            search = YelpAPI(self.yelpapi_api_key)
            res = search.business_match_query(**params)
        return res

    def get_params(self, name: str, address1: str, city: str, state:str, country='US') -> Dict[str, str]:
        """Get parameters for yelp API."""
        _params = {
            'name': name,
            'address1': address1,
            'city': city,
            'state': state,
            'country': country,
        }
        params = {**self.params, **_params}
        return params

    @staticmethod
    def _process_response(res: dict) -> str:
        """Process response from YelpAPI."""
        if "error" in res.keys():
            raise ValueError(f"Got error from YelpAPI: {res['error']}")
        return res['businesses'][0]['id']
