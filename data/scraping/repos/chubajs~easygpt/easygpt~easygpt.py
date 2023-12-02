"""
EasyGPT Module

This module defines the EasyGPT class for easier interaction with the OpenAI API.

Author: Sergey Bulaev
License: MIT
"""
from .version import version as __version__
import openai
from .gptmodels import GPTModel
import tiktoken
import logging
import time

# Set the OpenAI API key from environment variables


class EasyGPT:
    def __init__(self, openai, model_name, system_msg="", temperature=-1):
        """Initialize EasyGPT class with given model_name and optional system_msg and temperature."""
        self.openai = openai
        self.model = GPTModel(model_name)
        if system_msg == "":
            self.system_msg = self.model.get_system_message()
        else:
            self.system_msg = system_msg
        self.context = []
        if temperature != -1:
            self.model.set_temperature(temperature)

    def clear_context(self):
        """Clears the context history."""
        self.context = []

    def _compose_request(self, max_tokens, gpt_tokens, max_retries=1):
        """Internal utility to compose request for OpenAI ChatCompletion."""
        retries = 0      # Initial retry counter

        while retries <= max_retries:
            try:
                logging.info(f"Sending request to {self.model.get_name()} "
                             f"temp: {self.model.get_temperature()} "
                             f"tokens: {gpt_tokens} max tokens: {max_tokens}")

                return openai.ChatCompletion.create(
                    model=self.model.get_name(),
                    max_tokens=max_tokens,
                    n=1,
                    stop=self.model.get_stop(),
                    temperature=self.model.get_temperature(),
                    messages=self.context,
                )
            except openai.error.Timeout as e:
                logging.warning(f"Request timed out: {e}")
                retries += 1  # Increment the retry counter
                if retries <= max_retries:
                    logging.info("Retrying...")
                    time.sleep(2)  # Wait for 2 seconds before retrying
            except openai.error.InvalidRequestError as e:
                logging.error(f"Request failed: {e}")
                return None

        logging.error("Max retries reached. Request failed.")
        return None

    def process_response(self, response):
        """Process the OpenAI API response."""
        if response:
            assistant_message_dict = response.choices[0].message
            assistant_message = assistant_message_dict["content"]

            input_price = self.model.count_input_price(response.usage['prompt_tokens'])
            output_price = self.model.count_output_price(response.usage['completion_tokens'])

            return (assistant_message, input_price, output_price)
        else:
            return (None, None, None)

    def ask(self, question, max_tokens=0):
        """Main method for querying GPT-3 without a pre-set context."""
        self.create_context(question)
        gpt_tokens = self.num_tokens_from_messages(self.context, model=self.model.get_name())
        if max_tokens == 0:
            max_tokens = self.determine_max_tokens(max_tokens, gpt_tokens)
        response = self._compose_request(max_tokens, gpt_tokens)
        return self.process_response(response)

    def ask_with_context(self, context, max_tokens=0):
        """Main method for querying GPT-3 with a pre-set context."""
        self.context = context
        gpt_tokens = self.num_tokens_from_messages(self.context, model=self.model.get_name())
        max_tokens = self.determine_max_tokens(max_tokens, gpt_tokens)
        response = self._compose_request(max_tokens, gpt_tokens)
        return self.process_response(response)

    def determine_max_tokens(self, max_tokens, gpt_tokens):
        """Determines the appropriate max tokens for the query."""
        if max_tokens == 0:
            return self.model.get_max_tokens() - gpt_tokens
        return max_tokens

    def process_response(self, response):
        """Processes the GPT-3 API response."""
        assistant_message_dict = response.choices[0].message
        assistant_message = assistant_message_dict["content"]
        input_price = self.model.count_input_price(response.usage['prompt_tokens'])
        output_price = self.model.count_output_price(response.usage['completion_tokens'])
        return assistant_message, input_price, output_price

    def create_context(self, message):
        """Creates a new context or updates the existing context."""
        self.context = [{"role": "system", "content": self.system_msg}] if self.system_msg else []
        self.context.append({"role": "user", "content": message})

    def set_system_message(self, message):
        """Sets the system message."""
        self.system_msg = message

    def num_tokens_from_messages(self, messages, model="gpt-3"):
        """Calculates the number of tokens required for the message list."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        if model == "gpt-3.5-turbo":
            # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif model == "gpt-4":
            # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        elif model == "gpt-3.5-turbo-16k":
            # print("Warning: gpt-3.5-turbo-16k may change over time. Returning num tokens assuming gpt-3.5-turbo-16k-0301.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif model == "gpt-3.5-turbo-0613":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-4-0613":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                # print("checking " + str(key) + " : \n" + str(value))
                num_tokens += len(encoding.encode(value))
                # print(str(message)[40:] + " : " + str(len(encoding.encode(value))))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    
        return num_tokens


    def tokens_in_string(self, string):
        """Calculates the number of tokens in a string."""
        data = {
            "role": "user",
            "content": string
        }
        return self.num_tokens_from_messages([data], model=self.model.get_name())