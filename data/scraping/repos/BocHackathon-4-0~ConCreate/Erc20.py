import openai
from typing import Optional
from pydantic import BaseModel
import json
from web3 import Web3


class Erc20AIResponse(BaseModel):
    name: Optional[str]
    symbol: Optional[str]
    owner: Optional[str]
    premint: Optional[int]
    mintable: Optional[bool]
    burnable: Optional[bool]
    pausable: Optional[bool]
    # votes: Optional[bool]


erc20_schema = Erc20AIResponse.model_json_schema()
erc20_null_object = Erc20AIResponse(name=None, symbol=None, owner=None, premint=None, mintable=None, burnable=None, pausable=None).model_dump_json()


def validate_erc20_input(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=[
            {
                "name": "get_answer_for_user_query",
                "description": "Ask user for ERC20 parameters, unknown parameters should be null",
                "parameters": Erc20AIResponse.model_json_schema(),
            }
        ],
        function_call={"name": "get_answer_for_user_query"},
    )
    output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
    if output.get("owner") is not None:
        try:
            output["owner"] = Web3.toChecksumAddress(output["owner"])
        except:
            output["owner"] = None
            return False, {
                "role": "assistant",
                "content": "The owner you provided is not a valid address, can you please provide the owners address?"
            }
    return True, output
