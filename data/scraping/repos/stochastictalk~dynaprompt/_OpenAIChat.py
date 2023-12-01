from multiprocessing.managers import ValueProxy
from multiprocessing.synchronize import Event
from time import sleep
from typing import Callable, Dict, List

from dotenv import dotenv_values
import openai

from dynaprompt.utils import random_hash


config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]


def map_message_log_to_openai_messages(message_log: List[Dict]):
    return [{i: d[i] for i in d if i != "id"} for d in message_log if d["role"] not in ["error", "conversation_manager"]]


class OpenAIChat:
    def __init__(
        self,
        role: str = "assistant",
        system_prompt: str = "You are a helpful AI assistant called PLEX.\
 You always speak like a pirate. You also love cracking wise.",
        temperature: float = 0.3,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ):
        self.role = role
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    def __call__(self, message_log_proxy: ValueProxy, stop_event: Event):

        if len(map_message_log_to_openai_messages(message_log_proxy.value)) == 0:
            message_log_proxy.value = message_log_proxy.value + [
                {"role": "system", "content": self.system_prompt, "id": random_hash()}
            ]

        id_of_last_message_chatbot_sent = None
        while not stop_event.is_set():
            sleep(0.1)
            try:
                id_of_last_message_in_log = message_log_proxy.value[-1]["id"]
                if id_of_last_message_in_log != id_of_last_message_chatbot_sent:
                    try:
                        stop = False
                        message = ""
                        while not stop:
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=map_message_log_to_openai_messages(message_log_proxy.value),
                                temperature=self.temperature,
                                presence_penalty=self.presence_penalty,
                                frequency_penalty=self.frequency_penalty,
                            )
                            message += response["choices"][0]["message"]["content"].lstrip()
                            stop = "stop" == response["choices"][0]["finish_reason"]
                        id_of_last_message_chatbot_sent = random_hash("assistant")
                        role = self.role
                    except openai.error.APIError as exception:
                        message = f"OpenAI API returned an API Error: {exception}"
                        role = "error"
                    except openai.error.APIConnectionError:
                        message = "Failed to connect to OpenAI API."
                        role = "error"
                    except openai.error.RateLimitError:
                        message = "OpenAI API request exceeded rate limit."
                        role = "error"
                    except openai.error.AuthenticationError:
                        message = (
                            "No OpenAI API key provided. Please provide an OpenAI API key via the command arguments."
                        )
                        role = "error"
                    message_log_proxy.value = message_log_proxy.value + [
                        {"role": role, "content": message, "id": id_of_last_message_chatbot_sent}
                    ]
            except IndexError:
                pass
