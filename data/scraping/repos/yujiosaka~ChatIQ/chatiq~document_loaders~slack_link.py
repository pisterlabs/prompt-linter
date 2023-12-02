import logging
from datetime import datetime, timezone
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from chatiq.constants import FILE_DOCUMENT_THREAD_TS
from chatiq.text_processor import TextProcessor
from chatiq.utils import pretty_json_dumps

from .pdf import PdfLoader
from .plain_text import PlainTextLoader


class SlackLinkLoader(BaseLoader):
    """A class for loading attachments in Slack into Document objects.

    The SlackLinkLoader class is a specialized document loader that takes a Slack message and attachment
    and loads it into a Document object.
    """

    @classmethod
    def check_supported(cls, attachment: dict) -> bool:
        """Checks if the attachment is Slack link

        Returns:
            bool: If the attachment is Slack link or not
        """

        id = attachment.get("id")
        original_url = attachment.get("original_url")
        author = attachment.get("author_id")
        text = attachment.get("text")
        return id and original_url and author and text

    def __init__(self, body: dict, message: dict, attachment: dict, model: str):
        """Initializes the SlackLinkLoader with a Slack body.

        Args:
            body (dict): The Slack body to be loaded, represented as a dictionary.
            message (dict): The current message in the event body.
            attachment (dict): The attachment to the message.
            model (str): The name of the model used for tokenization.
        """

        self.logger = logging.getLogger(__name__)
        self.body = body
        self.message = message
        self.attachment = attachment
        self.processor = TextProcessor(model)

    def load(self) -> List[Document]:
        """Loads the Slack link into a Document object.

        Returns:
            List[Document]: A list containing a Document object.
        """

        if not self.check_supported(self.attachment):
            return []

        metadata = self._build_metadata()
        content = self.processor.truncate_text(self.attachment["text"])

        page_content = {
            "content_type": "slack_link",
            "user": self.message["user"],
            "author": self.attachment["author_id"],
            "channel": self.body["event"]["channel"],
            "content": content,
            "permalink": self.attachment["original_url"],
            "timestamp": metadata["timestamp"],
        }

        files = self._extract_files()
        if files:
            page_content["files"] = files

        return [Document(page_content=pretty_json_dumps(page_content), metadata=metadata)]

    def _build_metadata(self) -> dict:
        """Builds metadata for the Document object based on the attachment dictionary.

        Returns:
            dict: A dictionary containing metadata information for a Document.

        Raises:
            ValueError: If the file's timestamp cannot be properly converted to a datetime object.
        """

        event_time = self.body["event_time"]

        try:
            timestamp = datetime.fromtimestamp(event_time, timezone.utc)
        except ValueError:
            error_message = f"Error converting date: {event_time}"
            self.logger.error(error_message)
            raise ValueError(error_message)

        attachment_id = f"{self.message['ts']}-{self.attachment['id']}"

        return {
            "file_or_attachment_id": attachment_id,
            "content_type": "slack_link",
            "channel_type": self.body["event"]["channel_type"],
            "channel_id": self.body["event"]["channel"],
            "thread_ts": FILE_DOCUMENT_THREAD_TS,
            "ts": self.message["ts"],
            "timestamp": timestamp.isoformat(),
            "permalink": self.attachment["original_url"],
        }

    def _extract_files(self) -> List[dict]:
        """Extracts a list of file details in the Document object.

        Returns:
            List[dict]: A list of dictionaries containing the file details.
        """

        if not self.attachment.get("files"):
            return []

        return [
            {"title": file["title"], "permalink": file["permalink"]}
            for file in self.attachment["files"]
            if PlainTextLoader.check_supported(file) or PdfLoader.check_supported(file)
        ]
