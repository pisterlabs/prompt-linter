"""
Ai Assistant module
"""
import json
import os
from typing import Optional

import openai
from dotenv import load_dotenv


class AiAssistant:
    """
    AiAssistant based on the OpenAI ChatCompletion API.
    """

    __default_system_message = """

    You are an AI assistant.

    """

    __delimiter = "####"

    __coder_system_message = """

    You're a python developer and your task is to provide python code based on the prompt you receive from the user.

    The user query will be delimited with {} characters.

    Provide only code ready to be execuded. This code will be exececuted with the exec command by the user.
    Avoid any no-code content in your response.

    << FORMATTING >>
    Your response should be in JSON format. Here is the expected JSON format:
    [
     {{"code": "YOUR PYTHON CODE GOES HERE",
     "message": "any other comment you want to add to the response"}},
]

    While developing the code check the import of all the necessary libraries.

    This is a dictionary of all the available variables (vars()) with their own dtype: context= {} .
    If a variable is available in the context, do not instantiate it from scratch.

    Use the provided context to understand wich variables are available.
    You cannot access context in you code.

    Avoid any print or plot, if not specified by the user.

    """

    def __init__(
        self,
        system_message: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        if openai_api_key is None:
            if load_dotenv():
                openai.api_key = os.getenv("openai_api_key")
            elif "openai_api_key" in os.environ:
                openai.api_key = os.environ["openai_api_key"]
            else:
                raise ValueError("Missing OpenAI API key")
        else:
            openai.api_key = openai_api_key

        if system_message is None:
            self.system_message = self.__default_system_message
        else:
            self.system_message = system_message

        if model is None:
            self.model = "gpt-3.5-turbo"
        else:
            self.model = model

        self.temperature = temperature

    def reply(self, prompt: str) -> str:
        """
        :param prompt: ai assistant query
        :return: ai assistant reply

        Get an AI assistant reply from a user prompt
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        return response.choices[0].message.content

    def run_code(self, prompt: str, vars_: Optional[dict] = None, iterations: int = 5):
        """
        :param prompt: ai assistant query
        :param vars: context variables
        :param iterations: maximum code execution iterations
        :return: messages history

        Get an AI assistant reply from a user prompt
        """

        if vars_ is None:
            vars_ = {}

        messages = [
            {
                "role": "system",
                "content": self.__coder_system_message.format(
                    self.__delimiter, {key: type(vars_[key]) for key in vars_}
                ),
            },
            {
                "role": "user",
                "content": f"{self.__delimiter}{prompt}{self.__delimiter}",
            },
        ]

        response = openai.ChatCompletion.create(
            model=self.model, messages=messages, temperature=self.temperature
        )

        code = json.loads(response.choices[0].message.content)[0]["code"]

        messages.append({"role": "assistant", "content": code})

        for i in range(iterations):
            try:
                # pylint: disable=exec-used
                exec(code, globals(), vars_)
                # pylint: enable=exec-used
                break
            # pylint: disable=broad-exception-caught
            except Exception as error:
                # pylint: enable=broad-exception-caught
                print("------ Error in code execution ------")
                print(f"Error: {error}")
                print(f"Code: {code} \n \n")

                if i - 1 == iterations:
                    print("Maximum number of iterations reached")
                    break

                messages.append(
                    {
                        "role": "user",
                        "content": f"{self.__delimiter}I executed your response with 'exec' and \
                        I got this error: {str(error)}. \
                        Please answer with code only.{self.__delimiter}",
                    }
                )
                response = openai.ChatCompletion.create(
                    model=self.model, messages=messages, temperature=self.temperature
                )

                code = json.loads(response.choices[0].message.content)[0]["code"]

                messages.append({"role": "assistant", "content": code})

        return messages
