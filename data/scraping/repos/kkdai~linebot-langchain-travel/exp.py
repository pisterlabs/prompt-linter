import requests
import json

from langchain.tools import BaseTool
from langchain.agents import AgentType

from typing import Optional, Type
from pydantic import BaseModel, Field


class TravelExpInput(BaseModel):
    """Get the keyword about travel experience."""

    keyword: str = Field(...,
                         description="The city and state, e.g. San Francisco, CA")


class TravelExpTool(BaseTool):
    name = "search_experience"
    description = "Get the keyword about travel experience"

    def _run(self, keyword: str):
        exp_results = get_experience(keyword)
        return exp_results

    def _arun(self, keyword: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = TravelExpInput


def get_experience(keyword):
    api_url = "https://nextjs-chatgpt-plugin-starter.vercel.app/api/get-experience"
    headers = {'Content-Type': 'application/json'}

    # 根據API規範組建請求body
    data = {
        "keyword": keyword
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        return None
