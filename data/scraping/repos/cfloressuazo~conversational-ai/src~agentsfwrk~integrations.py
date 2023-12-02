import json
import os
import time
from typing import Union

import openai
from openai.error import APIConnectionError, APIError, RateLimitError

import agentsfwrk.logger as logger

log = logger.get_logger(__name__)

openai.api_key = os.getenv('OPENAI_API_KEY')

class OpenAIIntegrationService:
    def __init__(
        self,
        context: Union[str, dict],
        instruction: Union[str, dict]
    ) -> None:

        self.context = context
        self.instructions = instruction

        if isinstance(self.context, dict):
            self.messages = []
            self.messages.append(self.context)

        elif isinstance(self.context, str):
            self.messages = self.instructions + self.context

    def get_models(self):
        return openai.Model.list()

    def add_chat_history(self, messages: list):
        """
        Adds chat history to the conversation.
        """
        self.messages += messages

    def answer_to_prompt(self, model: str, prompt: str, **kwargs):
        """
        Collects prompts from user, appends to messages from the same conversation
        and return responses from the gpt models.
        """
        # Preserve the messages in the conversation
        self.messages.append(
            {
                'role': 'user',
                'content': prompt
            }
        )

        retry_exceptions = (APIError, APIConnectionError, RateLimitError)
        for _ in range(3):
            try:
                response = openai.ChatCompletion.create(
                    model       = model,
                    messages    = self.messages,
                    **kwargs
                )
                break
            except retry_exceptions as e:
                if _ == 2:
                    log.error(f"Last attempt failed, Exception occurred: {e}.")
                    return {
                        "answer": "Sorry, I'm having technical issues."
                    }
                retry_time = getattr(e, 'retry_after', 3)
                log.error(f"Exception occurred: {e}. Retrying in {retry_time} seconds...")
                time.sleep(retry_time)

        response_message = response.choices[0].message["content"]
        response_data = {"answer": response_message}
        self.messages.append(
            {
                'role': 'assistant',
                'content': response_message
            }
        )

        return response_data

    def answer_to_simple_prompt(self, model: str, prompt: str, **kwargs) -> dict:
        """
        Collects context and appends a prompt from a user and return response from
        the gpt model given an instruction.
        This method only allows one message exchange.
        """

        messages = self.messages + f"\n<Client>: {prompt} \n"

        retry_exceptions = (APIError, APIConnectionError, RateLimitError)
        for _ in range(3):
            try:
                response = openai.Completion.create(
                    model = model,
                    prompt = messages,
                    **kwargs
                )
                break
            except retry_exceptions as e:
                if _ == 2:
                    log.error(f"Last attempt failed, Exception occurred: {e}.")
                    return {
                        "intent": False,
                        "answer": "Sorry, I'm having technical issues."
                    }
                retry_time = getattr(e, 'retry_after', 3)
                log.error(f"Exception occurred: {e}. Retrying in {retry_time} seconds...")
                time.sleep(retry_time)

        response_message = response.choices[0].text

        try:
            response_data = json.loads(response_message)
            answer_text = response_data.get('answer')
            if answer_text is not None:
                self.messages = self.messages + f"\n<Client>: {prompt} \n" + f"<Agent>: {answer_text} \n"
            else:
                raise ValueError("The response from the model is not valid.")
        except ValueError as e:
            log.error(f"Error occurred while parsing response: {e}")
            log.error(f"Prompt from the user: {prompt}")
            log.error(f"Response from the model: {response_message}")
            log.info("Returning a safe response to the user.")
            response_data = {
                "intent": False,
                "answer": response_message
            }

        return response_data

    def verify_end_conversation(self):
        """
        Verify if the conversation has ended by checking the last message from the user
        and the last message from the assistant.
        """
        pass

    def verify_goal_conversation(self, model: str, **kwargs):
        """
        Verify if the conversation has reached the goal by checking the conversation history.
        Format the response as specified in the instructions.
        """
        messages = self.messages.copy()
        messages.append(self.instructions)

        retry_exceptions = (APIError, APIConnectionError, RateLimitError)
        for _ in range(3):
            try:
                response = openai.ChatCompletion.create(
                    model       = model,
                    messages    = messages,
                    **kwargs
                )
                break
            except retry_exceptions as e:
                if _ == 2:
                    log.error(f"Last attempt failed, Exception occurred: {e}.")
                    raise
                retry_time = getattr(e, 'retry_after', 3)
                log.error(f"Exception occurred: {e}. Retrying in {retry_time} seconds...")
                time.sleep(retry_time)

        response_message = response.choices[0].message["content"]
        try:
            response_data = json.loads(response_message)
            if response_data.get('summary') is None:
                raise ValueError("The response from the model is not valid. Missing summary.")
        except ValueError as e:
            log.error(f"Error occurred while parsing response: {e}")
            log.error(f"Response from the model: {response_message}")
            log.info("Returning a safe response to the user.")
            raise

        return response_data
