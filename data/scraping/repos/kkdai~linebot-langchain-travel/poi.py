import requests
import json

from langchain.tools import BaseTool
from langchain.agents import AgentType
from typing import Optional, Type
from pydantic import BaseModel, Field


class TravelPOIInput(BaseModel):
    """Get the keyword about travel information."""

    keyword: str = Field(...,
                         description="The city and state, e.g. San Francisco, CA")


class TravelPOITool(BaseTool):
    name = "search_poi"
    description = "Get the keyword about travel information"

    def _run(self, keyword: str):
        poi_results = get_pois(keyword)

        return poi_results

    def _arun(self, keyword: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = TravelPOIInput


def get_pois(keyword):
    """
    Query the get-poi API with the provided keyword.

    Parameters:
    keyword (str): The keyword for searching the position of interest.

    Returns:
    dict: The response from the API, should comply with getPoiResponse schema.
    """
    url = "https://nextjs-chatgpt-plugin-starter.vercel.app/api/get-poi"
    headers = {'Content-Type': 'application/json'}

    # The request data should comply with searchPoiRequest schema
    data = {"keyword": keyword}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}
