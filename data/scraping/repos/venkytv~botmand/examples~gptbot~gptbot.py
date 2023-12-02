#!/usr/bin/env python3

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator

from collections import deque
import logging
import os
import re
import sys
import tiktoken
import time
from typing import Deque, List

BOT_ID = os.environ["BOTMAND_USER_ID"]
BOT_NAME = os.environ["BOTMAND_USER_NAME"]

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
TEMPERATURE = int(os.environ.get("MODEL_TEMPERATURE", 0.6))

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 1000))
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", 100))   # 100 messages of history
MAX_AGE = int(os.environ.get("MAX_AGE", 60 * 30))       # 30 minutes of history
TONE = os.environ.get("TONE", "friendly")

VERBOSE = ( os.environ.get("VERBOSE", "") == "yes" )

logging.basicConfig(format="[%(levelname)s] %(message)s",
                    level=logging.DEBUG if VERBOSE else logging.INFO)

class ChatroomPromptTemplate(StringPromptTemplate, BaseModel):

    @validator("input_variables")
    def validate_input_variables(cls, v):
        if len(v) < 1 or "messages" not in v:
            raise ValueError("Input variables must be a list containing 'messages'.")
        return v

    def format(self, **kwargs) -> str:
        history = "\n".join(kwargs["messages"])

        prompt = f"""
        The following is a conversation between a set of humans and an AI chatbot called {BOT_NAME}.
        {BOT_NAME} is talkitive and provides lots of specific details from its context.
        If {BOT_NAME} does not know the answer to a question, it truthfully says so.
        Tone for {BOT_NAME}'s responses: {TONE}
        Format everything in markdown.

        Current conversation:
        {history}
        {BOT_NAME}: """

        return prompt

    def _prompt_type(self) -> str:
        return "chatroom-bot"

class Message:
    encoding = tiktoken.encoding_for_model(MODEL_NAME)

    def __init__(self, text: str):
        self.text = text
        self.timestamp = time.time()
        self.num_tokens = len(self.encoding.encode(text))

    @property
    def age(self) -> float:
        return time.time() - self.timestamp

class MessageBuffer:
    def __init__(self, max_length: int, max_tokens: int):
        self._messages: Deque[Message] = deque(maxlen=max_length)
        self.max_tokens = max_tokens
        self.num_tokens = 0

    def append(self, message: Message):
        self._messages.append(message)
        self.num_tokens += message.num_tokens

        while self.num_tokens > self.max_tokens:
            self.num_tokens -= self._messages.popleft().num_tokens

        # Delete older messages
        while self._messages and self._messages[0].age > MAX_AGE:
            self.num_tokens -= self._messages.popleft().num_tokens

    @property
    def messages(self) -> List[str]:
        return [m.text for m in self._messages if m.age < MAX_AGE]

    def length(self) -> int:
        return len(self._messages)

    def __str__(self):
        return f"N={self.length()} T={self.num_tokens}"

class ChatEngine:

    def __init__(self, bot_user_id, model_name=MODEL_NAME, temperature=TEMPERATURE,
                 max_history=MAX_HISTORY, max_tokens=MAX_TOKENS, verbose=False):
        self.bot_user_id = f"<@{bot_user_id}>"
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        self.llm = None
        self.chain = None

        self.buffer = MessageBuffer(max_length=max_history, max_tokens=max_tokens)
        self._openai_api_key = None

    @property
    def openai_api_key(self):
        if not self._openai_api_key:
            # Check if key is set in environment
            self._openai_api_key = os.environ.get("OPENAI_API_KEY", None)

        return self._openai_api_key

    def load_chain(self):
        api_key = self.openai_api_key
        if not api_key:
            return False

        self.llm = ChatOpenAI(openai_api_key=api_key, model_name=self.model_name, temperature=self.temperature)
        self.chain = LLMChain(llm=self.llm, prompt=ChatroomPromptTemplate(input_variables=["messages", "tone"]))

        return True

    # Record a message, and respond if necessary
    def record(self, raw_message):
        # Strip <> from usernames
        message = re.sub(r"<(U.*?)>", r"\1", raw_message.strip())

        # Replace bot user id with name
        message = message.replace(self.bot_user_id, BOT_NAME)

        # Add message to buffer
        self.buffer.append(Message(message))

        if self.bot_user_id in raw_message:
            # Directed message; respond
            self.respond("...")
            response = self.chat()

            if response:
                self.buffer.append(Message(f"{BOT_NAME}: {response}"))

                # Replace all usernames with slack mentions
                response = re.sub(r"(U.*?)\b", r"<@\1>", response)

                self.respond(response)

    # Run the chat engine and return a response
    def chat(self) -> str:
        if not self.chain:
            if not self.load_chain():
                self.respond(f"No OpenAI API key found in environment.\n" +
                             "Please set the OPENAI_API_KEY environment variable in the config.")
                sys.exit("No OpenAI API key found in environment.")
                return None

        try:
            return self.chain.run(messages=self.buffer.messages, tone=TONE)
        except Exception as e:
            return f"Error: {e}"

    # Print a response
    def respond(self, response):
        # Replace newlines with escaped newlines
        response = response.replace("\n", "\\n")
        print(response, flush=True)

    def shutdown(self):
        engine.respond("Shutting down...")

if __name__ == "__main__":
    logging.debug("Starting chat engine...")
    engine = ChatEngine(bot_user_id=BOT_ID, verbose=VERBOSE)
    try:
        for line in sys.stdin:
            engine.record(line)
    except KeyboardInterrupt:
        engine.shutdown()
