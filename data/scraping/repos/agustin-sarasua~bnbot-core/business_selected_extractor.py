from typing import List
from app.model import Message
import openai
import os
import json
from app.utils import logger
from datetime import datetime, timedelta

openai.api_key = os.environ.get('OPENAI_API_KEY')

system_message = """Save the bnbot_id of the business chosen by the user if the user has chosen one already.
Here are the available businesses:
{available_businesses}"""

json_fn = {
    "name": "set_business_selected",
    "description": "Saves the name of the business the user has chosen.",
    "parameters": {
        "type": "object",
        "properties": {
            "bnbot_id": {
                "type": "string",
                "description": "The bnbot_id of the business the user has chosen."
            },
        },
        "required": ["bnbot_id"]
    }
}

class BusinessSelectedExtractor:

    def run(self, messages: List[Message], available_businesses: str):

        formatted_system_message = system_message.format(available_businesses=available_businesses)

        messages_input = [{"role": "system", "content": formatted_system_message}]
        for msg in messages:
            messages_input.append({"role": msg.role, "content": msg.text})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages_input,
            functions=[json_fn],
            temperature=0., 
            max_tokens=500, 
        )
        if "function_call" in response.choices[0].message and "arguments" in response.choices[0].message["function_call"]:
            fn_parameters = json.loads(response.choices[0].message["function_call"]["arguments"])
            fn_parameters["user_has_selected"] = ("bnbot_id" in fn_parameters and fn_parameters["bnbot_id"] != "")
            logger.debug(f"set_business_selected fn_parameters {fn_parameters}")
            return fn_parameters
        
        return {"user_has_selected": False }