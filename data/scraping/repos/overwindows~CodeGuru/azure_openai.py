import openai
import os
import uuid
from typing import List, Dict
import src.constants as constants
import time
import logging


def create_system_message(system_message: str) -> Dict[str, str]:
    return {
        constants.ROLE: constants.SYSTEM,
        constants.CONTENT: system_message
    }


def create_query(query: str) -> Dict[str, str]:
    return {
        constants.ROLE: constants.USER,
        constants.CONTENT: query
    }


def create_response(response: str) -> Dict[str, str]:
    return {
        constants.ROLE: constants.ASSISTANT,
        constants.CONTENT: response
    }


class AzureOpenAISession:
    def __init__(
            self,
            system_message: str,
            examples: List[List[str]],
            stop: str
    ):
        # Creates a random session ID
        self.session_id = uuid.uuid4()

        # initializing examples and system message
        self.__system_message = system_message
        self.__examples = examples
        self.messages = self.__construct_init_messages()
        self.stop = stop

    def __construct_init_messages(self) -> List[Dict[str, str]]:
        messages_list = list()
        if self.__system_message:
            messages_list.append(create_system_message(self.__system_message))

        if self.__examples:
            for example in self.__examples:
                assert len(example) == 2
                messages_list.append(create_query(example[0]))
                messages_list.append(create_response(example[1]))

        return messages_list

    def append_query_and_response(self, query, response):
        self.messages.append(create_query(query))
        self.messages.append(create_response(response))


class AzureOpenAI:
    def __init__(
            self,
            engine: str = constants.DEFAULT_AZURE_OPENAI_ENGINE,
            max_tokens: int = 15000,
            top_p: float = 0.95,
            temperature: float = 0.7,
            frequency_penalty: float = 0,
            presence_penalty: float = 0,
            stop: str = None,
            max_retries: int = 10,
            retry_backoff_seconds: float = 60
    ):
        self.api_type = constants.AZURE
        self.api_base = os.getenv(constants.OPENAI_API_BASE)
        self.api_version = constants.DEFAULT_OPENAI_API_VERSION
        self.api_key = os.getenv(constants.OPENAI_API_KEY)

        # Completion parameters
        self.engine = engine
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

        # retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_retries = max_retries

        self.sessions = dict()

        self.initialize()

    def create_session(
            self,
            system_message: str = "You are an AI assistant that helps people find information.",
            examples: List[List[str]] = None,
            stop: str = None
    ) -> uuid:
        new_session = AzureOpenAISession(system_message=system_message, examples=examples, stop=stop)
        self.sessions[new_session.session_id] = new_session
        return new_session.session_id

    def ask_question(self, query: str = None, session_id: uuid = None) -> str:
        session = self.sessions.get(session_id, None)
        stop = None
        if session:
            stop = session.stop

        messages = AzureOpenAI.__construct_messages_with_session(query, session)
        response = None
        for try_cnt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    engine=self.engine,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=stop
                )

                break
            except openai.error.RateLimitError:
                logging.log(
                    logging.INFO,
                    f"Experienced rate limit error on try {try_cnt + 1}, retrying in {self.retry_backoff_seconds}s."
                )

                time.sleep(self.retry_backoff_seconds)

        if not response:
            raise "Unable to contact Azure Open AI endpoint!"

        response_content = response[constants.CHOICES][0][constants.MESSAGE][constants.CONTENT]
        if session and response_content:
            session.append_query_and_response(query, response_content)

        return response_content

    def initialize(self) -> None:
        openai.api_type = self.api_type
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        openai.api_version = self.api_version

    @staticmethod
    def __construct_messages_with_session(query: str = None, session: AzureOpenAISession = None):
        messages = list()
        if session:
            messages.extend(session.messages)

        if query is not None and len(query) > 0:
            messages.append(create_query(query))

        return messages
