import os

import openai
from typing import List, Dict
from AI.functions import get_function_list, execute_function
import json


class OpenAI:

    def __init__(self):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.functions = get_function_list()
        self.messages = [
            {
                'role': 'system',
                'content': 'You are Star the artificial intelligence assistant of the Nasa. Always use the function "get_information" to get the information. Act gentlemanly.'
            }
        ]

    def set_messages(self, messages: List[Dict[str, str]]):
        self.messages = messages

    def add_message(self, message: Dict[str, str]):
        self.messages.append(message)

    def generate_completion(self):
        function_call_end = True

        while function_call_end:
            function_call_end = False

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=self.messages,
                functions=self.functions,
                function_call="auto",
                max_tokens=500,
                stream=True
            )

            actual_content = ""
            actual_function_name = ""
            actual_function_arguments = ""

            for message in response:
                # Function call
                if "function_call" in message["choices"][0]["delta"]:
                    if "name" in message["choices"][0]["delta"]["function_call"]:
                        actual_function_name = message["choices"][0]["delta"]["function_call"]["name"]
                        actual_function_arguments = ""

                    if "arguments" in message["choices"][0]["delta"]["function_call"]:
                        actual_function_arguments += message["choices"][0]["delta"]["function_call"]["arguments"]

                # Content of the message
                if "content" in message["choices"][0]["delta"]:
                    if message["choices"][0]["delta"]["content"] is not None:
                        actual_content += message["choices"][0]["delta"]["content"]
                        yield str({"type": "message", "content": message["choices"][0]["delta"]["content"]})

                # Conversation end condition
                if message["choices"][0]["finish_reason"] is not None:
                    if message["choices"][0]["finish_reason"] == "function_call":
                        function_call_end = True
                    break

            if function_call_end:
                actual_function_arguments = json.loads(actual_function_arguments)

                execution = execute_function(actual_function_name, actual_function_arguments)

                response = None
                for i in execution:
                    if i['type'] == 'function':
                        yield str({'type': 'function', 'content': i['content']})
                    else:
                        response = {'type': 'message', 'content': i['content']}

                self.add_message({"role": "function", "name": actual_function_name, "content": str(response)})
            else:
                self.add_message({"role": "assistant", "content": actual_content})

    def __str__(self):
        return str(self.messages)

    def __repr__(self):
        return str(self.messages)
