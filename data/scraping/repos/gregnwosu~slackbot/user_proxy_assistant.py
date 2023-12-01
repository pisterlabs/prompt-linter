import subprocess
from enum import Enum
from pydantic import PrivateAttr
from typing import Literal
from instructor import OpenAISchema
from pydantic import Field

from slackbot.instruct import code_assistant
from slackbot.instruct.utils import get_completion
import slackbot.instruct.code_assistant
import os
from openai import OpenAI
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class SendMessage(OpenAISchema):
    """Send messages to other specialized agents in this group chat."""
    recepient: Literal['code_assistant'] = Field(...,
                                                 description="code_assistant is a world class programming AI capable of executing python code.")
    message: str = Field(...,
                         description="Specify the task required for the recipient agent to complete. Focus instead on clarifying what the task entails, rather than providing detailed instructions.")
    def run(self):
        agents_and_threads = {
            "code_assistant": {
                "agent": code_assistant.create(_client),
                "thread": None,
                "funcs": [code_assistant.File, code_assistant.ExecutePyFile]
            }
        }
        recipient = agents_and_threads[self.recepient]
        # if there is no thread between user proxy and this agent, create one
        if not recipient["thread"]:
            recipient["thread"] = _client.beta.threads.create()

        message = get_completion(message=self.message, **recipient)

        return message


def create(client: OpenAI):
    user_proxy = client.beta.assistants.create(
        name='User Proxy Agent',
        instructions="""As a user proxy agent, your responsibility is to streamline the dialogue between the user and specialized agents within this group chat.
    Your duty is to articulate user requests accurately to the relevant agents and maintain ongoing communication with them to guarantee the user's task is carried out to completion.
    Please do not respond to the user until the task is complete, an error has been reported by the relevant agent, or you are certain of your response.""",
        model="gpt-4-1106-preview",
        tools=[
            {"type": "function", "function": SendMessage.openai_schema},
        ],
    )
    return user_proxy
