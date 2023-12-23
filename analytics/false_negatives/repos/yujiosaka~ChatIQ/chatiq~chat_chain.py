from datetime import datetime
from typing import List

from dateutil import tz
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chains import RetrievalQA
from langchain.chat_models.base import BaseChatModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseRetriever
from langchain.tools import Tool

from chatiq.document_loaders import PdfLoader, PlainTextLoader, SlackLinkLoader, UnfurlingLinkLoader
from chatiq.prompt import (
    COMBINE_PROMPT_TEMPLATE,
    HUMAN_MESSAGE,
    INPUT_VARIABLES,
    QUESTION_PROMPT_TEMPLATE,
    SLACK_CONVERSATION_SEARCH_DESCRIPTION,
    SLACK_CONVERSATION_SEARCH_NAME,
    SLACK_URL_SEARCH_DESCRIPTION,
    SLACK_URL_SEARCH_NAME,
    SYSTEM_MESSAGE,
)
from chatiq.text_processor import TextProcessor
from chatiq.utils import pretty_json_dumps, utc_to_local


class ChatChain:
    """An orchestrator class for managing a chat conversation using a retrieval-based question-answering model.

    The ChatChain can fetch documents relevant to the user's query from an index. The chain
    runs this tool and adds a system message to the prompt during its operation.
    """

    def __init__(
        self,
        chat: BaseChatModel,
        memory: BaseChatMemory,
        retriever: BaseRetriever,
        bot_id: str,
        channel_id: str,
        context: str,
        timezone_offset: str,
    ):
        """Initializes a new instance of the ChatChain class.

        Args:
            chat (BaseChatModel): The language learning model.
            memory (BaseChatMemory): The chat memory.
            retriever (BaseRetriever): A retriever object for fetching documents from the index.
            bot_id (str): The ID of the bot.
            channel_id (str): The ID of the channel where the bot was mentioned.
            context (str): The context of the team configuration.
            timezone_offset (str): The timezone_offset of the team configuration.
        """

        self.chat = chat
        self.memory = memory
        self.retriever = retriever
        self.bot_id = bot_id
        self.channel_id = channel_id
        self.context = context
        self.timezone_offset = timezone_offset

        self.processor = TextProcessor(self.chat.model_name)
        self.qa = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="map_reduce",
            chain_type_kwargs={
                "question_prompt": QUESTION_PROMPT_TEMPLATE,
                "combine_prompt": COMBINE_PROMPT_TEMPLATE,
            },
            retriever=retriever,
        )
        self.tools = [
            Tool(name=SLACK_CONVERSATION_SEARCH_NAME, func=self.qa.run, description=SLACK_CONVERSATION_SEARCH_DESCRIPTION),
            Tool(name=SLACK_URL_SEARCH_NAME, func=self.retriever.search_url, description=SLACK_URL_SEARCH_DESCRIPTION),
        ]
        self.chain = initialize_agent(
            self.tools,
            self.chat,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            agent_kwargs={
                "system_message": SYSTEM_MESSAGE,
                "human_message": HUMAN_MESSAGE,
                "input_variables": INPUT_VARIABLES,
            },
            memory=self.memory,
        )

    def add_memory_ai_message(self, message: dict) -> None:
        """Add an AI message to the memory.

        Args:
            message (dict): The message to be added to the memory.
        """

        self.memory.chat_memory.add_ai_message(self._format_message(message, with_timestamp=True))

    def add_memory_user_message(self, message: dict) -> None:
        """Add a user message to the memory.

        Args:
            message (dict): The message to be added to the memory.
        """

        self.memory.chat_memory.add_user_message(self._format_message(message, with_timestamp=True))

    def run(self, message: dict) -> str:
        """Run the question-answering tool on a user's input and return the response.

        The method passes the user's input to the question-answering tool and returns its response.

        Args:
           message (dict): The last message as the input.

        Returns:
            str: The response from the question-answering tool.
        """

        utc_now = datetime.now(tz=tz.UTC)
        if self.timezone_offset == "+00:00":
            time_message = f"Current time is '{utc_now.isoformat()}'. "
        else:
            local_now = utc_to_local(utc_now, self.timezone_offset)
            time_message = f"Current local time is '{local_now.isoformat()}'. Respect local timezone by default. "

        input = f"Human: {self._format_message(message)}"
        context = self.context or "Not set"

        return self.chain.run(
            input=input,
            bot_id=self.bot_id,
            channel_id=self.channel_id,
            time_message=time_message,
            context=context,
        )

    def _format_message(self, message: dict, with_timestamp: bool = False) -> str:
        """Formats a message. Truncates the text if it exceeds the maximum token length of the chat model.

        Args:
            message (dict): The message to be formatted.
            with_timestamp (bool, optional): If True, adds a timestamp to the formatted message. Defaults to False.

        Returns:
            str: The formatted message.
        """

        text = self.processor.truncate_text(message["text"])
        content = {"user_id": message["user"], "action": "Message", "action_input": text}

        if with_timestamp:
            utc_datetime = datetime.fromtimestamp(float(message["ts"]))
            local_datetime = utc_to_local(utc_datetime, self.timezone_offset)
            content["timestamp"] = local_datetime.isoformat()

        unfurling_links = self._extract_unfurling_links(message)
        if unfurling_links:
            content["unfurling_links"] = unfurling_links

        slack_links = self._extract_slack_links(message)
        if slack_links:
            content["slack_links"] = slack_links

        files = self._extract_files(message)
        if files:
            content["files"] = files

        return pretty_json_dumps(content)

    def _extract_unfurling_links(self, message: dict) -> List[dict]:
        """Extracts a list of attachment details from a message.

        Args:
            message (dict): The message to to extract unfurling links.

        Returns:
            List[dict]: A list of dictionaries containing the unfurling link details.
        """

        if not message.get("attachments"):
            return []

        return [
            {
                "title": self.processor.truncate_text(attachment["title"], length=100),
                "permalink": self.processor.truncate_text(attachment["original_url"], length=100),
            }
            for attachment in message["attachments"]
            if UnfurlingLinkLoader.check_supported(attachment)
        ]

    def _extract_slack_links(self, message: dict) -> List[dict]:
        """Extracts a list of attachment details from a message.

        Args:
            message (dict): The message to to extract Slack links.

        Returns:
            List[dict]: A list of dictionaries containing the Slack link details.
        """

        if not message.get("attachments"):
            return []

        return [
            {
                "author": attachment["author_id"],
                "content": self.processor.truncate_text(attachment["text"], length=100),
                "permalink": self.processor.truncate_text(attachment["original_url"], length=100),
            }
            for attachment in message["attachments"]
            if SlackLinkLoader.check_supported(attachment)
        ]

    def _extract_files(self, message: dict) -> List[dict]:
        """Extracts a list of file details from a message.

        Args:
            message (dict): The message to to extract unfurling.

        Returns:
            List[dict]: A list of dictionaries containing the file details.
        """

        if not message.get("files"):
            return []

        return [
            {"title": file["title"], "permalink": file["permalink"]}
            for file in message["files"]
            if PlainTextLoader.check_supported(file) or PdfLoader.check_supported(file)
        ]
