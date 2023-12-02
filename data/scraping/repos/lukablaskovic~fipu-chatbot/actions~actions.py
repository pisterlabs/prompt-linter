import os
import openai
import json

from dotenv import load_dotenv
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import time

from .db import fipuAPI, get_internship_details, get_available_companies

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class ActionShowData(Action):
    def name(self) -> Text:
        return "action_show_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        data = fipuAPI().fetch_data()

        readable = [(item["id"], item["name"]) for item in data]

        message = "\n".join([f"ID: {item[0]}, Name: {item[1]}" for item in readable])

        dispatcher.utter_message(
            text=f"Izvolite listu dostupnih poduzeća za obavljanje prakse:\n\n{message}"
        )

        return [SlotSet("results", readable)]


class ActionDetail(Action):
    def name(self) -> Text:
        return "action_get_details"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        question = tracker.latest_message["text"]
        # Step 1: send the conversation and available functions to GPT
        messages = [{"role": "user", "content": question}]

        functions = [
            {
                "name": "get_internship_details",
                "description": "Get the details about the available internship tasks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "The ID of the company you want to get the details for",
                        },
                        "name": {
                            "type": "string",
                            "description": "The name of the company you want to get details for, might not be exact.",
                            "enum": get_available_companies(),
                        },
                    },
                    "required": ["id"] or ["name"],
                },
            }
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=functions,
            function_call="auto",  # default
        )
        reply_conent = response["choices"][0]
        print(reply_conent)
        response_message = reply_conent["message"]

        # check if GPT wanted to call a function
        if response_message.get("function_call"):
            # Step 3: call the function
            available_functions = {
                "get_internship_details": get_internship_details,
            }  # only one function in this example, but you can have multiple
            function_name = response_message["function_call"]["name"]
            print("\n***function_name*** ", function_name)
            fuction_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            print("\n***function_args*** ", function_args)
            function_response = fuction_to_call(
                id=function_args.get("id"),
                name=function_args.get("name"),
            )
            print("\n***function_response*** ", json.dumps(function_response))
            print("\n***response_message*** ", response_message)
            # Step 4: send the info on the function call and function response to GPT

            messages.append(
                {
                    "role": "system",
                    "content": "You are a helpful assistant, provide all the details about the available internship tasks from JSON object. Answer in Croatian language.",
                }
            )
            messages.append(
                response_message
            )  # extend conversation with assistant's reply
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
            )  # extend conversation with function response
            print("\n***messages to send to GPT*** ", json.dumps(messages, indent=4))
        try:
            time.sleep(5)
            second_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
            )  # get a new response from GPT where it can see the function response
            message_response = second_response.choices[0]["message"]["content"]
            dispatcher.utter_message(message_response)
        except Exception as e:
            print(f"Error during second GPT call: {e}")
            dispatcher.utter_message(text=f"Error: {e}")
