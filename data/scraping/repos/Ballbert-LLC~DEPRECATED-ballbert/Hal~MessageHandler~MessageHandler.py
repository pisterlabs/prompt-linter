import json
from typing import Generator
import openai
import websockets

from Hal import Assistant
from ..Utils import get_functions_list, generate_system_message
from ..Logging import log_line
from Config import Config
from ..Classes import Response

config = Config()


class MessageHandler:
    def __init__(self, assistant: Assistant, gpt_response) -> None:
        self.function_name = ""
        self.arguments = ""
        self.full_message = ""
        self.assistant = assistant
        self.gpt_response = gpt_response

        openai.api_key = config["OPENAI_API_KEY"]

    def get_functions(self, message):
        relevant_ids = self.assistant.pm.get_relevant(message)
        relevant_actions = {
            key: self.assistant.action_dict[key]
            for key in relevant_ids
            if key in self.assistant.action_dict
        }
        functions = get_functions_list(relevant_actions)

        return functions

    def add_to_messages(self, message):
        self.assistant.messages.append({"role": "user", "content": message})

    def add_function_to_messages(self, message, function_name):
        self.assistant.messages.append(
            {"role": "function", "name": function_name, "content": str(message)}
        )

    def ask_gpt(self, functions):
        log_line(f"I: {self.assistant.messages[-1]}")

        return openai.ChatCompletion.create(
            model=config["LLM"],
            messages=[generate_system_message(), *self.assistant.messages],
            temperature=config["TEMPATURE"],
            functions=functions,
            stream=True,
        )

    def handle_chunk(self, chunk):
        delta = chunk["choices"][0]["delta"]
        # check for end
        if delta == {}:
            if self.function_name:
                self.assistant.messages.append(
                    {
                        "role": "assistant",
                        "function_call": {
                            "arguments": str(self.arguments),
                            "name": str(self.function_name),
                        },
                        "content": "",
                    }
                )

                if self.function_name:
                    try:
                        self.arguments = json.loads(self.arguments)
                    except Exception as e:
                        log_line("Err", e)
                        self.arguments = {}

                    log_line(f"FN: {self.function_name}")
                    log_line(f"Arg: {self.arguments}")

                    try:
                        function_result: Response = self.assistant.action_dict[
                            self.function_name
                        ]["function"](**self.arguments)
                    except Exception as e:
                        log_line("function error", e)

                    if function_result.suceeded:
                        function_result = function_result.data
                    else:
                        function_result = "error" + function_result.data
                        log_line("function found error: ", function_result)

                    function_message_handler = MessageHandler(
                        self.assistant, self.gpt_response
                    )

                    return function_message_handler.handle_function(
                        function_result, self.function_name
                    )
            else:
                self.assistant.messages.append(
                    {"role": "assistant", "content": self.full_message}
                )
                log_line(f"A: {self.full_message}")

        if "function_call" in delta:
            function_call = delta["function_call"]
            if "name" in function_call:
                self.function_name = function_call["name"]
            elif "arguments" in function_call:
                self.arguments += function_call["arguments"]
        elif "content" in delta:
            self.full_message += delta["content"]
            return delta["content"]

        return None

    def handle_function(self, message, function_name):
        self.add_function_to_messages(message, function_name)
        functions = self.get_functions(
            f"{self.gpt_response}, {function_name}:{message}"
        )
        current_chunk = ""
        res = self.ask_gpt(functions)
        for chunk in res:
            chunk_result = self.handle_chunk(chunk)
            if isinstance(chunk_result, Generator):
                for item in self.handle_generatior(chunk_result):
                    current_chunk = item
                    yield item
            if chunk_result:
                current_chunk = chunk_result
                yield chunk_result

        if isinstance(current_chunk, str):
            if not current_chunk[-1] in ".?!'":
                yield "."

    def handle_generator(self, generator):
        for item in generator:
            if isinstance(item, Generator):
                for sub_item in self.handle_generator(item):
                    yield sub_item
            else:
                yield item

    async def handle_message(self):
        async with websockets.connect("ws://localhost:8765") as websocket:

            async def send_to_websocket(item, is_final):
                json_data = json.dumps(
                    {"type": "assistant", "message": item, "is_final": is_final}
                )

                await websocket.send(json_data)

            async def send_user_to_ws():
                json_data = json.dumps({"type": "user", "message": self.gpt_response})

                await websocket.send(json_data)

            async def send_color_to_ws():
                json_data = json.dumps({"type": "color", "color": "green"})

                await websocket.send(json_data)

            await send_color_to_ws()
            await send_user_to_ws()

            try:
                self.add_to_messages(self.gpt_response)
                functions = self.get_functions(self.gpt_response)
                res = self.ask_gpt(functions)

                current_chunk = ""

                for chunk in res:
                    chunk_result = self.handle_chunk(chunk)
                    if isinstance(chunk_result, Generator):
                        for item in self.handle_generator(chunk_result):
                            current_chunk = item
                            yield item
                            await send_to_websocket(item, False)
                    elif chunk_result:
                        current_chunk = chunk_result
                        yield chunk_result
                        await send_to_websocket(chunk_result, False)

                if isinstance(current_chunk, str):
                    if current_chunk and not current_chunk[-1] in ".?!'":
                        yield "."
                await send_to_websocket("", True)

            except Exception as e:
                log_line("err", e)
                yield "I got an error please try again"
                await send_to_websocket("", True)
                return
