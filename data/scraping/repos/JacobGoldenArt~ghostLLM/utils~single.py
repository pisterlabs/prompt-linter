# === File: handle_feedback.py ===
import logging
from typing import List, Dict, Tuple, Optional
from rich.table import Table
from rich import print as rprint

error_logger = logging.getLogger("error_logger")

#Define a new type for messages
Message = Dict[str, str]
#Define a new type for the history
History = List[Message]

class LLMfeedback:
    """A class for formatting any errors or logging output from the LLM or User Settings"""

    @staticmethod
    def provide_feedback(msg: str) -> None:
        """Provide feedback based on verbose mode."""
        rprint(f"[bold blue]{msg}[/bold blue]")

    @staticmethod
    def log_and_handle_errors(e: BaseException, verbose: Optional[str], user_input: str, full_history: History, msg: Optional[str]=None) -> None:
        """Log errors and handle them based on verbose mode."""
        error_logger.error("An error occurred: %s", e)
        if msg:
            # print the extra message if provided
            rprint(f"[bold red]{msg}[/bold red]")
        LLMfeedback.handle_verbose_output(verbose, user_input=user_input, full_history=full_history)

    @staticmethod
    def handle_verbose_output(verbose: Optional[str], user_input: str, full_history: History, msg: Optional[str]=None) -> None:
        """Handle verbose output based on verbose mode."""
        if verbose is None:
            return
        if verbose == "obj":
            rprint("User Input:", user_input)
            if msg:
                # print the extra message if provided
                rprint(f"[bold blue]{msg}[/bold blue]")
        elif verbose == "thread":
            LLMfeedback.print_verbose_output(full_history)

    @staticmethod
    def print_verbose_output(full_history: History) -> None:
        table = Table(title="API Response")
        table.add_column("Role")
        table.add_column("Content")

        for message in full_history:
            role, content = message["role"], message["content"]
            role_style, content_style = LLMfeedback.get_styles_for_role(role)
            table.add_row(f"[{role_style}]{role}[/]", f"[{content_style}]{content}[/]")

            rprint(table)

    @staticmethod
    def get_styles_for_role(role: str) -> Tuple[str, str]:
        if role == "system":
            return "bold yellow", "yellow"
        elif role == "user":
            return "bold green", "green"
        else:  # role is 'assistant'
            return "bold blue", "blue"


# === File: history.py ===
from typing import Dict, List

class History:
    """A class for maintaining the history of a conversation."""

    def __init__(self, sys: str) -> None:
        self.history = [{"role": "system", "content": sys}]

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_last(self, n: int) -> List[Dict[str, str]]:
        return self.history[-n:]

    def get_full_history(self) -> List[Dict[str, str]]:
        return self.history

# === File: chat_history.py ===
from typing import Optional, Dict, List, Union
from bson.objectid import ObjectId
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from logs.handle_feedback import LLMfeedback
import os, datetime
import dotenv

# Load the .env file
dotenv.load_dotenv()

uri = os.environ.get("MONGO_URI")

class History:
    """A class for maintaining and storing the history of a conversation."""
    def __init__(self, sys: str, verbose: Optional[str] = None) -> None:
        """
        Initialize a conversation with a system message and setup connection to MongoDB.
        sys: str : The initial system message.
        """
        self.client = MongoClient(uri, server_api=ServerApi("1"))
        self.db = self.client.ghost_db
        self.chat_history_db = self.db.chat_history
        self.history = [{"role": "system", "content": sys}]
        self.verbose = verbose

        # Create a session at startup and save it to get session_id
        session_data = {'turns': self.history, 'time_stamp': datetime.datetime.now()}
        session_id = self.chat_history_db.insert_one(session_data).inserted_id
        self.session_id = session_id  # keeping track of session ID

        # Ping the MongoDB to check if connected
        try:
            self.client.admin.command('ping')
            LLMfeedback.provide_feedback("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            LLMfeedback.log_and_handle_errors(e, self.verbose, user_input="", full_history=[], msg="Connection to MongoDB failed!")

    def add(self, role: str, content: str):
        """
        Append a new role-content pair to the chatbot history and update the database.
        role : str : The role related to this content, either 'user' or 'assistant'.
        content : str : The content for this role.
        """
        self.history.append({"role": role, "content": content})
        self.save_to_db()
        LLMfeedback.provide_feedback("Chat added to the db")

    def save_to_db(self):
        """
        Update the existing document in the MongoDB with the modified chat history.
        """
        self.chat_history_db.update_one({'_id': self.session_id}, {'$set': {'turns': self.history}})

    def load_from_db(self, session_id: str) -> None:
        # Load a session based on session_id

        # Clear current history first - as we would be loading a previous session
        self.history = []

        # MongoDB find_one returns None if the document does not exist
        session_data = self.chat_history_db.find_one({'_id': ObjectId(session_id)})
        if session_data:
            self.history = session_data.get('turns', [])
            # Set the current session_id to the loaded session's ID
            self.session_id = session_id
        else:
            print(f"No session found with provided id: {session_id}")

    def get_last(self, n: int) -> List[Dict[str, str]]:
        return self.history[-n:]

    def get_full_history(self) -> List[Dict[str, str]]:
        return self.history

    def __del__(self):
        # Close the client connection when the instance is deleted
        self.client.close()

# === File: openai_base.py ===
import os
from typing import Dict, List, Optional, Union
import openai
from openai import ChatCompletion
import dotenv
from logs.handle_feedback import LLMfeedback
from memory.chat_history import History

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenaiAPIBase:
    """Base class for OpenAI API calls."""

    def __init__(self, sys: str, verbose: Optional[str] = None, temperature: float = 0.5, max_tokens: int = 200, model: str = "gpt-3.5-turbo") -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.verbose = verbose
        self.history = History(sys, verbose)

    def make_api_call(self, last_turns: List[Dict[str, str]], additional_params: Optional[Dict[str, Union[str, int, float, List[Dict[str, str]]]]] = None) -> ChatCompletion:
        """Make an API call to OpenAI."""
        default_params = {
            "model": self.model,
            "messages": last_turns,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if additional_params:
            default_params.update(additional_params)

        api_response = None
        try:
            api_response = openai.ChatCompletion.create(**default_params)
            LLMfeedback.handle_verbose_output(self.verbose, user_input=last_turns[-1]["content"], full_history=self.history.get_full_history())

        except openai.APIError as e:
            # Handle API error here, e.g. retry or log
            LLMfeedback.log_and_handle_errors(e, self.verbose, last_turns[-1]["content"], self.history.get_full_history())

        return api_response

# === File: openai_chat.py ===
from typing import List, Dict, cast
from openai_api.openai_base import OpenaiAPIBase

class OpenaiChatCompletion(OpenaiAPIBase):
    """A class for interacting with the OpenAI Chat API."""

    def __call__(self, user_input: str) -> str:
        self.history.add("user", user_input)
        last_turns = self.history.get_last(5)

        response = self.make_api_call(last_turns)
        bot_response = ""
        if response:
            choices = cast(List[Dict[str, Dict[str, str]]], response["choices"])
            bot_response = choices[0]["message"]["content"]
            self.history.add("assistant", bot_response)

        return bot_response

# === File: main.py ===
from rich.markdown import Markdown
from rich import print as rprint
from openai_api.openai_chat import OpenaiChatCompletion

# Usage with verbose mode on, "obj" for object or "thread" for full thread
my_chat = OpenaiChatCompletion(
    sys="You are a helpful assistant.", verbose="thread", temperature=0.6, max_tokens=30
)

while True:
    user_input = input("Write message here: ")
    chat_response = my_chat(user_input)
    rprint("[bold cyan]DarkMatterBot:[/bold cyan]")
    rprint(Markdown(chat_response))
