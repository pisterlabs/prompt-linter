import logging
from datetime import datetime, timezone
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TokenTextSplitter

from chatiq.constants import FILE_DOCUMENT_THREAD_TS
from chatiq.text_processor import TextProcessor
from chatiq.utils import pretty_json_dumps


class PlainTextLoader(BaseLoader):
    PLAIN_TEXT_FILETYPES = {
        "text",
        "applescript",
        "c",
        "csharp",
        "cpp",
        "css",
        "csv",
        "clojure",
        "coffeescript",
        "d",
        "dart",
        "diff",
        "dockerfile",
        "docs",
        "erlang",
        "fortran",
        "go",
        "gpres",
        "groovy",
        "gzip",
        "html",
        "handlebars",
        "haskell",
        "haxe",
        "java",
        "javascript",
        "json",
        "kotlin",
        "latex",
        "lisp",
        "lua",
        "markdown",
        "matlab",
        "mumps",
        "objc",
        "ocaml",
        "pascal",
        "perl",
        "php",
        "pig",
        "powershell",
        "puppet",
        "python",
        "r",
        "ruby",
        "rust",
        "sql",
        "sass",
        "scala",
        "scheme",
        "shell",
        "smalltalk",
        "swift",
        "tsv",
        "vb",
        "vbscript",
        "velocity",
        "xml",
    }

    """A class for loading plain text files in Slack into Document objects.

    The PlainTextLoader class is a specialized document loader that takes a Slack message and file
    and loads it into a Document object.
    """

    @classmethod
    def check_supported(cls, file: dict) -> bool:
        """Checks if the file is plain text

        Returns:
            bool: If the file is plain text or not
        """

        return file.get("filetype") in cls.PLAIN_TEXT_FILETYPES

    def __init__(self, content: str, body: dict, file: dict, channel_type: str, model: str):
        """Initializes the PlainTextLoader with web client and a Slack body.

        Args:
            content (str): The content to the file.
            body (dict): The Slack body to be loaded, represented as a dictionary.
            file (dict): The file attached to the message.
            channel_type (str): Type of the channel (channel, group, im or mpim)
            model (str): The name of the model used for tokenization.
        """

        token_length = TextProcessor.get_token_length_for_model(model)

        self.splitter = TokenTextSplitter(model_name=model, chunk_size=token_length)
        self.logger = logging.getLogger(__name__)
        self.content = content
        self.body = body
        self.file = file
        self.channel_type = channel_type

    def load(self) -> List[Document]:
        """Loads the Slack event into a Document object.

        Returns:
            List[Document]: A list containing a Document object.
        """

        if not self.check_supported(self.file):
            return []

        metadata = self._build_metadata()

        texts = self.splitter.split_text(self.content)
        page_count = len(texts)

        documents = []
        for i, text in enumerate(texts):
            page_content = pretty_json_dumps(
                {
                    "content_type": self.file["filetype"],
                    "user": self.body["event"]["user_id"],
                    "name": self.file["name"],
                    "title": self.file["title"],
                    "channel": self.body["event"]["channel_id"],
                    "content": text,
                    "page": f"{i + 1} / {page_count}",
                    "permalink": self.file["permalink"],
                    "timestamp": metadata["timestamp"],
                }
            )
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents

    def _build_metadata(self) -> dict:
        """Builds metadata for the Document object based on the file dictionary.

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

        return {
            "file_or_attachment_id": self.file["id"],
            "content_type": self.file["filetype"],
            "channel_type": self.channel_type,
            "channel_id": self.body["event"]["channel_id"],
            "thread_ts": FILE_DOCUMENT_THREAD_TS,
            "ts": self.body["event"]["event_ts"],
            "timestamp": timestamp.isoformat(),
            "permalink": self.file["permalink"],
        }
