import json
import logging

import openai
import tiktoken
from tenacity import (  # for exponential backoff
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from repo_gpt.openai_service import num_tokens_from_messages, num_tokens_from_string


class MemoryStore:
    summary_prompt = """*Briefly* summarize this partial conversation about programming.
    Include less detail about older parts and more detail about the most recent messages.
    Start a new paragraph every time the topic changes!

    This is only part of a longer conversation so *DO NOT* conclude the summary with language like "Finally, ...". Because the conversation continues after the summary.
    The summary *MUST* include the function names, libraries, packages that are being discussed.
    The summary *MUST* include the filenames that are being referenced!
    The summaries *MUST NOT* include ```...``` fenced code blocks!

    Phrase the summary with the USER in first person, telling the ASSISTANT about the conversation.
    Write *as* the user.
    The user should refer to the assistant as *you*.
    Start the summary with "I asked you...".
    """

    SUMMARY_MODEL = "gpt-3.5-turbo-16k-0613"

    def __init__(
        self,
        system_prompt,
        user_task,
        functions=[],
        threshold=4000,
        summary_model=SUMMARY_MODEL,
    ):
        self.messages = []
        self.threshold = threshold
        self.summary_model = summary_model
        self.system_prompt = system_prompt
        self.user_task = user_task
        self._initialize_messages()
        self.functions = functions

    def _initialize_messages(self):
        initial_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_task},
        ]
        self.messages = initial_messages

    def add_message(self, message):
        self.messages.append(message)
        if self._count_messages_tokens() >= self.threshold:
            self.compress_messages()

    def get_messages(self):
        return self.messages

    def _count_messages_tokens(self):
        return num_tokens_from_messages(
            self.messages, "gpt-4"
        ) + num_tokens_from_string(json.dumps(self.functions), "gpt-4")

    # @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))

    def get_formatted_messages(self):
        # output = StringIO()
        # pprint.pprint(self.messages, stream=output)
        # formatted_messages = output.getvalue()

        # formatted_messages = json.dumps(self.messages)
        formatted_messages = []
        for message in self.messages:
            if "function_call" in message:
                # Message to call a function
                formatted_messages.append(
                    f"calling function {message['function_call']['name']}({str(message['function_call']['arguments'])})"
                )
            elif "name" in message:
                # Message with function results
                formatted_messages.append(
                    f"function {message['name']} returned: {message['content']}"
                )
            else:
                formatted_messages.append(f"{message['role']}: {message['content']}")

        return "\n".join(formatted_messages)
        # return "test"

    def compress_messages(self):
        # TODO: use something intelligent like semantic search possibly to select relevant messages

        summary_messages = [
            {
                "role": "system",
                "content": f"You are an expert technical writer.",
            },
            {
                "role": "user",
                "content": f"{self.summary_prompt}\n{self.get_formatted_messages()}",
            },
        ]
        try:
            response = openai.ChatCompletion.create(
                model=self.SUMMARY_MODEL, messages=summary_messages
            )
            logging.debug(response)
            assistant_message = response["choices"][0]["message"]
            logging.debug(assistant_message)
            self._initialize_messages()
            assistant_message.role = "user"
            self.messages.append(assistant_message)
            return response
        except Exception as e:
            logging.error("Unable to generate ChatCompletion response")
            logging.error(f"Exception: {e}")
            raise
