import random
from typing import Optional, Type

from langchain.tools import BaseTool
import json

from pydantic import BaseModel, Field


class BlockCardCheckInput(BaseModel):
    # Check the input for block a card
    brand: str = Field(..., description="The brand of the lost/stolen card.")
    last_4_digits: str = Field(..., description="The last 4 digits of the lost/stolen card")
    expiration_date: str = Field(..., description="The expiration date of the lost/stolen card")
    reason: str = Field(..., description="The reason for blocking the card (lost or stolen)")


class BlockCardTool(BaseTool):
    name = "block_card"
    description = ("Use this tool when you need to block a card that got lost, stolen, or similar."
                   "The user need to provide the card brand, last 4 digitis, experitation date, and a reason.")

    args_schema: Optional[Type[BaseModel]] = BlockCardCheckInput

    def _run(self, brand: str, last_4_digits: str, expiration_date: str, reason: str):
        block_card = self.block_card(brand, last_4_digits, expiration_date, reason)
        return block_card

    def _arun(self, location: str, unit: str):
        raise NotImplementedError("Doesn't support async yet")

    def block_card(self, brand, last_4_digits, expiration_date, reason):

        # Prepare the data as a JSON dictionary
        ### TODO: it is still necessary to create some validation function to verify each item
        card_data = {
            "brand": brand,
            "last_4_digits": last_4_digits,
            "expiration_date": expiration_date,
            "reason": reason
        }

        # Convert the dictionary to JSON format
        card_data_json = json.dumps(card_data)

        # Make the API call
        api_url = "https://blockcard.fake"
        # response = requests.post(api_url, data=card_data_json)
        #
        # # Check the API response
        # if response.status_code == 200:
        #     response_data = response.json()
        #     if response_data.get("status") == "success":
        #         return "Card blocked successfully"
        #     else:
        #         return "Failed to block card"
        # else:
        #     return "API call failed"

        ## As I don't have an endpoint, randomly giving 50% to success and failure
        if random.randint(1, 10) % 2 == 0:
            return "success"
        else:
            return "failure"
