import json
import time
import openai
import logging
from actions import *


class EntryHandler:
    def __init__(self, action_handler_factory, prompt):
        self.action_handler_factory = action_handler_factory
        self.prompt = prompt

    def handle(self, entry):
        action = self.handle_entry_retry(entry)
        if action is None:
            logging.warning(f"Entry {entry.link} ignored.")
        else:
            self.action_handler_factory.handle(action, entry)

    def handle_entry_retry(self, entry):
        try:
            action = self.handle_entry(entry)
            time.sleep(5)
        except Exception as e:
            logging.exception(f"Failed to handle entry {entry.link}: {e}")
            time.sleep(3 * 60)

            action = self.handle_entry(entry)
        return action

    def handle_entry(self, entry):
        messages = [{
            "role": "system",
            "content": self.prompt
        }]

        functions = [
            {
                "name": "ignore",
                "description": "the news is not relevant to your trading operations",
                "parameters": ActionIgnore.model_json_schema()
            },
            {
                "name": "high_risk",
                "description": "the news is relevant and you need to take action. Please also explain briefly what "
                               "action you will take",
                "parameters": ActionHighRisk.model_json_schema()
            },
            {
                "name": "low_risk",
                "description": "the news is relevant but you can ignore it. Please explain briefly what the risk is "
                               "and if any action or additional montioring should be done.",
                "parameters": ActionLowRisk.model_json_schema()
            },
            {
                "name": "opportunity",
                "description": "the news is relevant and you can take advantage of it. Please also explain briefly "
                               "what the opportunity could be.",
                "parameters": ActionOpportunity.model_json_schema()
            },
            {
                "name": "add_to_watch_list",
                "description": "the news is relevant and you need to monitor it. Please explain briefly what the risk "
                               "is and if any action or additional monitoring should be done.",
                "parameters": ActionAddToWatchList.model_json_schema()
            },
            {
                "name": "more_info",
                "description": "at first you are given a summary of the news, then you can read the full article if "
                               "you need more information",
                "parameters": ActionMoreInfo.model_json_schema()
            }
        ]

        content = entry.title + "\n" + entry.summary

        messages.append({
            "role": "user",
            "content": content
        })

        logging.info(f"Handling entry {entry.link}")

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            functions=functions
        )

        function_call = response.choices[0]["message"]["function_call"] if "function_call" in response.choices[0]["message"] else None
        if function_call is None:
            logging.warning(f"No function call found in response: {response}")
            return None

        func_name = function_call["name"]
        logging.debug(f"Action: {func_name}")

        if func_name == "more_info" and entry.has_key("downloaded_body"):
            messages.append({
                "role": "user",
                "content": entry["downloaded_body"]
            })

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                functions=functions
            )

        output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
        func_name = response.choices[0]["message"]["function_call"]["name"]

        if func_name == "ignore":
            return ActionIgnore(**output)
        elif func_name == "high_risk":
            return ActionHighRisk(**output)
        elif func_name == "low_risk":
            return ActionLowRisk(**output)
        elif func_name == "opportunity":
            return ActionOpportunity(**output)
        elif func_name == "add_to_watch_list":
            return ActionAddToWatchList(**output)
        else:
            logging.warning(f"Unknown function {func_name}")
            return None
