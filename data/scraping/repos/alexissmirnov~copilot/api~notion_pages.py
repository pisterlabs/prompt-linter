"""Notion page loader for langchain"""

from typing import Any, Dict, Optional


from langchain.docstore.document import Document
from langchain.document_loaders.notiondb import NotionDBLoader

NOTION_BASE_URL = "https://api.notion.com/v1"
DATABASE_URL = NOTION_BASE_URL + "/databases/{database_id}/query"
PAGE_URL = NOTION_BASE_URL + "/pages/{page_id}"
BLOCK_URL = NOTION_BASE_URL + "/blocks/{block_id}/children"


class NotionPageLoader(NotionDBLoader):
    """page loader"""
    def __init__(
        self,
        integration_token: str,
        request_timeout_sec: Optional[int] = 10
    ) -> None:
        super().__init__(integration_token=integration_token,
                         database_id='None', request_timeout_sec=request_timeout_sec)

    def load_page_by_id(self, page_id) -> Document:
        """ load_page_by_id """
        metadata: Dict[str, Any] = {}
        metadata["id"] = page_id

        data = self._request(PAGE_URL.format(page_id=page_id))
        metadata["title"] = data["properties"]["Title"]["title"][0]["text"]["content"]

        return Document(page_content=self._load_blocks(page_id), metadata=metadata)
