import os

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.utils import spinning
from IPython.display import Markdown
import textwrap

from .util import (
    spinning
)


# Other imports and classes should be here, like GitHubConnector, ToPrint, ToGithubIssue

class Bot:
    pass

class TrIAge(Bot):
    # ... (The rest of the class definition remains the same)

    def __init__(
        self, 
        model_provider,
        model_api_key,
        hub_api_key, 
        model_name="gpt-3.5-turbo",
        channel=ToPrint(),
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self.set_api_key(model_api_key)
        self.hub = GitHubConnector(hub_api_key)
        self.channel = channel
        
        self.chat = ChatOpenAI(
            temperature=0,
            model_name=self.model_name,
        )
        self._configure()

    def set_api_key(self, api_key):
        """ Set the API key for the model provider."""
        if self.model_provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            raise NotImplementedError(f"unknown model provider {self.model_provider}")

    # ... (The rest of the class methods remain the same)

    @spinning(text="Configuring...")
    def _configure(self):
        # ... (The rest of the _configure method remains the same)

        self.chat_history = [
            SystemMessage(content=self.mission),
        ]

        for job in self.jobs:
            self.chat_history.append(
                SystemMessage(content=job),
            )

        for instruction in self.instructions:
            self.chat_history.append(
                SystemMessage(content=instruction),
            )

        self.chat(self.chat_history)

    # ... (The rest of the class methods remain the same)

    @spinning(text="Thinking...")
    def tell_system(
        self,
        prompt,
    ):
        self.chat_history.append(SystemMessage(content=prompt))
        response = self.chat(self.chat_history)
        self.display(response.content)

    @spinning(text="Thinking...")
    def tell(
        self,
        prompt,
    ):
        self.chat_history.append(HumanMessage(content=prompt))
        response = self.chat(self.chat_history)
        self.chat_history.append(response)
        self.display(response.content)
        return response

    # ... (The rest of the class methods remain the same)
