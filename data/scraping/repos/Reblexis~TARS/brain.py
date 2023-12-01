import json
from typing import Any

import openai
from abc import ABC, abstractmethod

from constants import *
from DataManagement.file_system import ensure_open_ai_api
from Function.Core.command_parameters import *


class BrainController(ABC):
    """
    This class is the abstract class representing the brain of the agent. It should be responsible for processing
    the information from its senses and deciding what to do next. In the future this will also contain the long term
    memory.
    """
    def __init__(self, cm):
        self.command_manager = cm
        pass

    @abstractmethod
    def process(self, query: dict) -> dict:
        pass


class GPT3BrainController(BrainController):
    """
    This class represents a possible implementation of the brain of the agent. It uses the GPT-3 language model to generate
    responses to the user's speech. It also uses the command manager to execute commands.
    """

    BEHAVIOUR_PROMPT = ("You are controlling a robot. You are responding to a content recognized from user's speech. You can answer with keyword pass if "
                        "you think that the robot should ignore the user's speech or if you have finished your response."
                        " You have available commands which you can use that control the robot. Anything you say (excluding 'pass') will be said by the robot."
                        "Remember you HAVE to use the keyword pass to finish your response.")

    SAVED_MESSAGES_COUNT = 10
    INITIAL_MESSAGE_COUNT = 5

    ASSISTANT_STREAK_LIMIT = 3
    MAX_TOKENS = 300

    def __init__(self, cm):
        super().__init__(cm)
        self.functions = None
        ensure_open_ai_api()
        self.messages = []
        self.functions = self.get_commands_gpt3()
        self.initialize_messages()

    def initialize_messages(self):
        self.messages = [
            {"role": "system", "content": self.BEHAVIOUR_PROMPT},
            {"role": "user", "content": "Assistant. Hello how are you?"},
            {"role": "assistant", "content": "I'm doing well thank you. Pass."},
            {"role": "user", "content": "Assistant. Where"},
            {"role": "assistant", "content": "Pass."},
        ]

    def delete_old_messages(self):
        if len(self.messages) <= self.SAVED_MESSAGES_COUNT + self.INITIAL_MESSAGE_COUNT:
            return

        messages_backup = self.messages.copy()
        self.initialize_messages()
        for message in messages_backup[-self.SAVED_MESSAGES_COUNT:]:
            self.messages.append(message)

    def get_commands_gpt3(self) -> list:
        functions = []
        for cmd_name, (func, cmd_class) in self.command_manager.commands.items():
            parameters = {"type": "object", "properties": {}, "required": []}
            for parameter in cmd_class().parameters:
                if isinstance(parameter, ContinuousParameter):
                    parameters["properties"][parameter.name] = {"type": "number", "description": parameter.description}
                    parameters["required"].append(parameter.name)
                elif isinstance(parameter, DiscreteParameter):
                    parameters["properties"][parameter.name] = {"type": "string", "description": parameter.description,
                                                                "enum": parameter.options}
                    parameters["required"].append(parameter.name)
                elif isinstance(parameter, TextParameter):
                    parameters["properties"][parameter.name] = {"type": "string", "description": parameter.description}
                    parameters["required"].append(parameter.name)
            functions.append({
                "name": cmd_name,
                "description": cmd_class().description,
                "parameters": parameters
            })
        return functions

    def execute_command_gpt3(self, command_info: dict) -> str:
        print(f"Executing command {command_info}")

        command_name: str = command_info["name"]
        command_args: dict = json.loads(command_info["arguments"])
        feedback_message: str = self.command_manager.commands[command_name][0](command_args)
        if feedback_message is None:
            feedback_message = "None"
        return feedback_message

    def get_response(self, recursion_counter: int = 0) -> list:
        self.delete_old_messages()

        if recursion_counter == self.ASSISTANT_STREAK_LIMIT-1:
            self.messages.append({"role": "system", "content": "You can send only one more message."
                                                               " You have to respond to the user now and"
                                                               " finish with 'pass'."})

        if recursion_counter == self.ASSISTANT_STREAK_LIMIT:
            return [{"content": "This is a scripted message. I don't know what to do."}]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            temperature=0,
            messages=self.messages,
            functions=self.functions,
            function_call='auto',
            max_tokens=self.MAX_TOKENS,
        )
        response_message = response["choices"][0]["message"]
        response_content = response_message["content"]
        if response_content is None:
            response_content = ""

        self.messages.append(response_message)
        if response_message.get("function_call"):
            function_feedback: str = self.execute_command_gpt3(response_message.get("function_call"))
            self.messages.append({"role": "function", "name": response_message["function_call"]["name"],
                                  "content": function_feedback})

        self.command_manager.say({"text": response_content.replace("Pass", "").replace("pass", "").strip()})

        assistant_responses = [response_content]

        if "pass" in response_content or "Pass" in response_content:
            return assistant_responses

        assistant_responses.extend(self.get_response(recursion_counter + 1))
        return assistant_responses

    def process(self, query: dict) -> list:
        if "spoken_content" not in query:
            return [{"content": "pass"}]

        spoken_content: str = query["spoken_content"]
        self.messages.append({"role": "user", "content": spoken_content})
        responses: list = self.get_response()
        print("Assistant responses:")
        print(responses)
        print("------------------")
        print("Messages:")
        print(self.messages)
        print("------------------")

        return responses


if __name__ == "__main__":
    command_manager = CommandManager(None)
    bc = GPT3BrainController(command_manager)
    print(bc.process({"spoken_content": "Please turn on the camera"}))
