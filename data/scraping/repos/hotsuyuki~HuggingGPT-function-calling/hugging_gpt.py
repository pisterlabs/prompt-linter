import dotenv
import json
import openai
import os
import termcolor

from . import huggingface_available_functions


DEBUG_PRINT_COLOR = "blue"
MAX_NUM_CONVERSATION = 5


class HuggingGPT:
    def __init__(self, model: str = "gpt-3.5-turbo", is_verbose: bool = False) -> None:
        print(termcolor.colored("HuggingGPT.__init__()", DEBUG_PRINT_COLOR))
        print()

        self.model = model
        self.is_verbose = is_verbose
        self.system_content = ""

        dotenv.load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

        openai.api_key = OPENAI_API_KEY
        self.huggingface_available_functions_instance = \
            huggingface_available_functions.HuggingfaceAvailableFunctions(HUGGINGFACE_API_KEY, is_verbose=self.is_verbose)

        if self.is_verbose:
            print(termcolor.colored("OPENAI_API_KEY = " + OPENAI_API_KEY[:8] + "...", DEBUG_PRINT_COLOR))
            print(termcolor.colored("HUGGINGFACE_API_KEY = " + HUGGINGFACE_API_KEY[:8] + "...", DEBUG_PRINT_COLOR))
            print()

    def run(self, user_content: str) -> tuple[str, str]:
        print(termcolor.colored("HuggingGPT.run()", DEBUG_PRINT_COLOR))
        print()

        messages = [
            {"role": "system", "content": self.system_content},
            {"role": "user", "content": user_content}
        ]
        response_message = {}

        used_function_name = ""
        num_conversation = 0
        is_stop = False

        while not is_stop:
            # In order to save the amount of input tokens (prompt_tokens),
            # adds `functions` argument only if the last role is `user`.
            # https://platform.openai.com/docs/guides/gpt/function-calling
            if messages[-1]["role"] == "user":
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    functions=self.huggingface_available_functions_instance.get_available_functions_list()
                )
            else:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                )

            response_message = response["choices"][0]["message"]

            if self.is_verbose:
                print(termcolor.colored(f"{messages = }", DEBUG_PRINT_COLOR))
                print()
                print(termcolor.colored(f"{response = }", DEBUG_PRINT_COLOR))
                print()

            if response_message.get("function_call"):
                function_name = response_message["function_call"]["name"]
                used_function_name = function_name

                function_arguments = response_message["function_call"].get("arguments")
                if function_arguments and isinstance(function_arguments, str):
                    function_arguments = json.loads(function_arguments)

                function_content = self.huggingface_available_functions_instance.call_function(
                    function_name,
                    function_arguments,
                    user_content
                )

                if self.is_verbose:
                    print(termcolor.colored(f"{function_name = }", DEBUG_PRINT_COLOR))
                    print(termcolor.colored(f"{function_arguments = }", DEBUG_PRINT_COLOR))
                    print()
                    print(termcolor.colored(f"{function_content = }", DEBUG_PRINT_COLOR))
                    print()

                response_message = response_message.to_dict()
                response_message["function_call"] = response_message["function_call"].to_dict()
                messages.append(response_message)

                messages.append({"role": "function", "name": function_name, "content": function_content})

            num_conversation += 1
            is_stop = (response["choices"][0]["finish_reason"] == "stop") or (MAX_NUM_CONVERSATION <= num_conversation)

        assert response_message["role"] == "assistant"
        assistant_content = response_message["content"]

        return assistant_content, used_function_name
