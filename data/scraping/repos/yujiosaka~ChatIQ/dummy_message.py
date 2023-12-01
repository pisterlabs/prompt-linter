import time
from datetime import datetime, timezone
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from chatiq.utils import pretty_json_dumps


class DummyMessageLoader(BaseLoader):
    """A class for dummy Document objects.

    LangChain throws index out of range error when vectorstore has no document to query against.
    The DummyMessageLoader class is a specialized document loader to workaround the issue.
    """

    EVENT_ID = "DUMMY"
    USER_ID = "DUMMY"
    CHANNEL_ID = "DUMMY"
    MESSAGE = "Do not use this doucment to answer questions."

    def load(self) -> List[Document]:
        """Load dummy Document objects.

        Returns:
            List[Document]: A list containing a dummy Document object.
        """

        metadata = self._build_metadata()

        return [
            Document(
                page_content=pretty_json_dumps(
                    {
                        "content_type": "message",
                        "user": self.USER_ID,
                        "channel": self.CHANNEL_ID,
                        "message": self.MESSAGE,
                        "permalink": metadata["permalink"],
                        "timestamp": metadata["timestamp"],
                    }
                ),
                metadata=metadata,
            )
        ]

    def _build_metadata(self) -> dict:
        """Builds the dummy metadata for the Document object.

        Returns:
            dict: A dictionary containing the dummy metadata.
        """

        timestamp = time.time()

        return {
            "file_or_attachment_id": self.EVENT_ID,
            "content_type": "message",
            "channel_type": "channel",
            "channel_id": self.CHANNEL_ID,
            "thread_ts": str(timestamp),
            "ts": str(timestamp),
            "timestamp": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "permalink": f"https://matomel.slack.com/archives/{self.CHANNEL_ID}/{timestamp * 1000000}",
        }
