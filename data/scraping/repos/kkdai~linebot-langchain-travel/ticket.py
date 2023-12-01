import json
import requests
from langchain.tools import BaseTool
from langchain.agents import AgentType

from typing import Optional, Type
from pydantic import BaseModel, Field


class SearchTicketsInput(BaseModel):
    """Get the travel ticket information."""

    fromCity: str = Field(...,
                          description="The airport name from where the aircraft is scheduled to depart.")
    toCity: str = Field(...,
                        description="The destination or the airport name where the aircraft is scheduled to arrive.")
    startDate: str = Field(...,
                           description="The scheduled time at which the aircraft is set to depart.")
    backDate: str = Field(...,
                          description="The scheduled time at which the aircraft is expected to return or arrive back.")
    numOfAdult: int = Field(...,
                            description="Total count of adult passengers for the flight.")


class TravelTicketTool(BaseTool):
    name = "search_tickets"
    description = "Get the travel ticket information"

    def _run(self, fromCity: str, toCity: str, startDate: str, backDate: str, numOfAdult: int):
        ticket_results = get_ticket(
            fromCity, toCity, startDate, backDate, numOfAdult)
        return ticket_results

    def _arun(self, fromCity: str, toCity: str, startDate: str, backDate: str, numOfAdult: int):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = SearchTicketsInput


def get_ticket(fromCity, toCity, startDate, backDate, numOfAdult):
    api_url = "https://nextjs-chatgpt-plugin-starter.vercel.app/api/get-ticket"
    headers = {'Content-Type': 'application/json'}

    # 根據API規範組建請求body
    data = {
        "fromCity": fromCity,
        "toCity": toCity,
        "startDate": startDate,
        "backDate": backDate,
        "numOfAdult": numOfAdult
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        return None
