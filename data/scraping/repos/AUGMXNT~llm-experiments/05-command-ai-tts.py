#!/usr/bin/env python

'''
Based on https://github.com/amachino/command-ai
- Sentence-based TTS grafted in, now w/ threaded worker for smoother TTS.
- TODO: whisper.cpp?
'''

import argparse
from   collections import deque
from   datetime import datetime
import json
import numpy as np
import openai
import os
from   os import path
import readline
import re
import sounddevice as sd
import sys
from   sys import stdout
import time
import threading
from   TTS.api import TTS
from   typing import NamedTuple, Optional


class VoiceWorker(threading.Thread):
    def __init__(self):
        super(VoiceWorker, self).__init__()
        
        # Silent Load of TTS
        print("Loading TTS...", end=' ')
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        model_name = 'tts_models/en/vctk/vits'
        self.speaker = 'p273'
        self.tts = TTS(model_name, progress_bar=False)

        sys.stdout.close()
        sys.stdout = original_stdout

        sd.default.samplerate = 22050
        print("DONE")

        self.lines = deque()
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            if self.lines:
                line = self.lines.popleft()
                
                # Play Item
                if line.strip():
                    original_stdout = sys.stdout
                    sys.stdout = open(os.devnull, 'w')
                    wav = self.tts.tts(line.strip(), speaker=self.speaker)
                    sys.stdout.close()
                    sys.stdout = original_stdout

                    sd.play(wav)
                    sd.wait()
            else:
                # Wait for more lines
                time.sleep(0.05)

    def stop(self):
        self.stop_event.set()
        

class ChatCompletionParams(NamedTuple):
    """
    Parameters for the chat completion API.

    See API docs for details
    https://platform.openai.com/docs/api-reference/chat
    """
    model: str = "gpt-3.5-turbo"
    messages: list = []
    max_tokens: int = 1000
    temperature: float = 1.0
    top_p: float = None
    n: int = 1
    stream: bool = True
    # stop: list = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict = {}
    user: str = ""


class ChatConfig(NamedTuple):
    context: str = ""
    params: ChatCompletionParams = ChatCompletionParams()


class ChatService:
    """
    A chat service that uses the OpenAI API to generate responses.

    Attributes
    ----------
    context : str
        The context for the conversation.
    history : ChatHistory
        The conversation history.
    params : ChatCompletionParams
        The parameters for the chat completion API.
    """

    def __init__(
        self,
        config=ChatConfig(),
    ):
        """
        Parameters
        ----------
        config : ChatConfig
            The configuration for the chat service.
        """
        self.context = config.context
        self.params = config.params

        self.history = ChatHistory(context=config.context)

        # Initialize Speaker
        self.speaker = VoiceWorker()
        self.speaker.start()


    def start(self) -> None:
        """Start the chat service."""
        print('Type "/help" to see available commands.\n')

        try:
            while True:
                line = input(">>> ")
                if line == "":
                    continue
                if line == "/help":
                    self.handle_command_help()
                    continue
                if line == "/log":
                    self.handle_command_log()
                    continue
                if line == "/save":
                    self.handle_command_save()
                    continue
                if line == "/clear":
                    self.handle_command_clear()
                    continue
                if line == "/forget":
                    self.handle_command_forget()
                    continue
                if line == "/context":
                    self.handle_command_context()
                    continue
                if line == "/exit":
                    break

                self.stream_completion(line, self.params)

        except (KeyboardInterrupt, EOFError):
            pass
        except openai.error.InvalidRequestError as e:
            print(e)
        except Exception:
            raise
        finally:
            self.speaker.stop()
            self.speaker.join()
            print("\nGoodbye!\n")

    def handle_command_help(self) -> None:
        """Print the help message."""
        print(
            """
/help    - view available commands
/exit    - exit the program
/log     - view the current conversation log
/save    - save the conversation log to a file
/clear   - clear all the conversation log
/forget  - cancel the previous message
/context - show the current chat context
"""
        )

    def handle_command_log(self) -> None:
        """Print the conversation log."""
        log = self.history.get_log()
        if log:
            print(f"\n\033[96m{log}\033[00m\n")
        else:
            print("\nempty\n")

    def handle_command_save(self) -> None:
        """Save the conversation log to a file."""
        file_path = self.history.save_log()
        print("\nsaved: " + file_path + "\n")

    def handle_command_clear(self) -> None:
        """Clear all the history of the conversation."""
        self.history.clear_log()
        print("\ncleared\n")

    def handle_command_forget(self) -> None:
        """Delete the last message in the conversation."""
        self.history.remove_last_conversation()
        previous_message = self.history.get_last_message()
        if previous_message:
            print("\n" + previous_message["content"] + "\n")

    def handle_command_context(self) -> None:
        """Show the current context."""
        print("\n" + self.context + "\n")

    def create_prompt_messages(self, line: str) -> list:
        """Create a prompt for the AI to complete."""
        messages = []
        context = self.context.strip()
        if context:
            messages.append({"role": "system", "content": context})
        log = self.history.get_messages()
        if log:
            messages.extend(log)
        prompt = {"role": "user", "content": line}
        messages.append(prompt)
        self.history.add_message(prompt)
        return messages

    def stream_completion(self, line: str, params: ChatCompletionParams) -> None:
        """Stream completions to stdout as they become available."""
        messages = self.create_prompt_messages(line)

        params = params._replace(messages=messages, stream=True, n=1)

        stream = openai.ChatCompletion.create(**params._asdict())

        stdout.write("\n")
        buf = ""
        sentence = ""
        for obj in stream:
            choice = obj.choices[0]
            delta = choice["delta"]
            if "content" in delta:
                content = delta["content"]
                # don't print initial empty lines
                if buf != "" or not content.strip() == "":
                    stdout.write(content)
                    stdout.flush()
                    buf += content

                    # Speak each sentence
                    sentence += content
                    if content.strip() in {".", "!", "?"} or content[-1] == "\n":
                        self.speaker.lines.append(sentence)
                        sentence = ""
            
        # Output the rest:
        if sentence:
            self.speaker.lines.append(sentence)

        buf = buf.strip()
        
        stdout.write("\n\n")
        self.history.add_message({"role": "assistant", "content": buf})


class ChatHistory:
    """
    The history of the conversation.

    Attributes
    ----------
    context : str
        The context for the conversation.
    messages : list[dict]
        The messages in the conversation.
    """

    def __init__(self, context=""):
        self.context = context
        self.messages: list = []

    def add_message(self, message: dict) -> None:
        """Add a new message to the conversation."""
        self.messages.append(message)

    def get_messages(self) -> list:
        """Return the messages in the conversation."""
        return self.messages

    def get_last_message(self) -> Optional[dict]:
        """Return the last message in the conversation."""
        return self.messages[-1] if self.messages else None

    def remove_last_conversation(self) -> None:
        """Remove the last message in the conversation."""
        if self.messages:
            self.messages = self.messages[:-2]

    def get_log(self) -> str:
        """Return the text of the conversation log."""
        log = ""
        for message in self.messages:
            if message["role"] == "user":
                log += ">>> " + message["content"].strip() + "\n\n"
            if message["role"] == "assistant":
                log += message["content"].strip() + "\n\n"
        log = log.strip()
        return log

    def save_log(self) -> str:
        """Save the conversation log to a file and return the path."""
        home_dir = path.expanduser("~")
        log_dir = path.join(home_dir, ".ai", "log")
        if not path.exists(log_dir):
            os.makedirs(log_dir)
        now = datetime.now()
        file_name = now.strftime("%Y%m%d%H%M%S") + ".jsonl"
        file_path = path.join(log_dir, file_name)

        with open(file_path, mode="w") as f:
            context = {"role": "system", "content": self.context}
            log = json.dumps(context, ensure_ascii=False) + "\n"
            log += "\n".join([json.dumps(m, ensure_ascii=False) for m in self.messages])
            f.write(log)
        return file_path

    def clear_log(self) -> None:
        """Clear the conversation log."""
        self.messages.clear()


def read_args() -> argparse.Namespace:
    """Create a CompletionParams object from command line arguments."""
    default = ChatCompletionParams()

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", default=default.model)
    parser.add_argument("-M", "--max_tokens", type=int, default=default.max_tokens)
    parser.add_argument("-t", "--temperature", type=float, default=default.temperature)

    return parser.parse_args()


def create_chat_config() -> ChatConfig:
    context = load_context()

    args = read_args()
    params = ChatCompletionParams(**vars(args))

    config = ChatConfig(context=context, params=params)
    return config


def load_context() -> str:
    """Read the chat context from `~/.ai/context.txt` and return as a string."""
    home_dir = path.expanduser("~")
    context_file = path.join(home_dir, ".ai", "context.txt")

    context = ""

    now = datetime.now()
    current_time = now.strftime("%a, %b %d %Y %I:%M %p")
    timezone = now.astimezone().tzname()
    context += f"Current time: {current_time} {timezone}\n\n"

    if path.exists(context_file):
        with open(context_file, mode="r") as f:
            context += f.read().strip()
    else:
        context += f"The following is a conversation with an AI assistant named Eva. The assistant is helpful, creative, clever, and very friendly." 
    return context


def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Please set OPENAI_API_KEY environment variable.")
        exit()

    config = create_chat_config()
    chat = ChatService(config=config)
    chat.start()


if __name__ == "__main__":
    main()
