"""Notion DB loader for langchain"""
import itertools
import time
from typing import Any, Dict, List
import requests

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.base import BaseLoader

NOTION_BASE_URL = "https://api.notion.com/v1"
DATABASE_URL = NOTION_BASE_URL + "/databases/{database_id}/query"
PAGE_URL = NOTION_BASE_URL + "/pages/{page_id}"
BLOCK_URL = NOTION_BASE_URL + "/blocks/{block_id}/children"
TIMEOUT = 10000
WAIT = 2
RETRY_COUNT = 3
METADATA_FILTER = ['id', 'title', 'tags', 'version', 'source id', 'published', 'source']


def metadata_filter(pair: tuple) -> bool:
    key, value = pair
    if key in METADATA_FILTER:
        return True
    else:
        return False


def _get_pdf_content(url_str: str, page_id: str) -> List[Document]:
    if url_str.startswith("http"):
        loader = PyPDFLoader(url_str)
        # loader = OnlinePDFLoader(url_str)
        pages = loader.load()
        return pages
    raise ValueError(f"Invalid URL of pdf: '{url_str}' at page_id: '{page_id}'")


class MyNotionDBLoader(BaseLoader):
    """Notion DB Loader.
    Reads content from pages within a Noton Database.
    Args:
        integration_token (str): Notion integration token.
        database_id (str): Notion database id.
    """

    def __init__(self, integration_token: str, database_id: str) -> None:
        """Initialize with parameters."""
        if not integration_token:
            raise ValueError("integration_token must be provided")
        if not database_id:
            raise ValueError("database_id must be provided")

        self.token = integration_token
        self.database_id = database_id
        self.headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    def load(self) -> List[Document]:
        """Load documents from the Notion database.
        Returns:
            List[Document]: List of documents.
        """
        page_ids = self._retrieve_page_ids()
        return list(itertools.chain.from_iterable(self.load_page(page_id) for page_id in page_ids))

    def _retrieve_page_ids(
            self, query_dict: Dict[str, Any] = {"page_size": 100}
    ) -> List[str]:
        """Get all the pages from a Notion database."""
        pages: List[Dict[str, Any]] = []
        query_dict = {
            "filter": {
                "and": [
                    {
                        "property": "Pub",
                        "checkbox": {
                            "equals": True
                        }
                    },
                    {
                        "property": "Status",
                        "select": {
                            "does_not_equal": "Published"
                        }
                    }
                ]
            },
            'page_size': 100
        }

        while True:
            data = self._request(
                DATABASE_URL.format(database_id=self.database_id),
                method="POST",
                query_dict=query_dict,
            )

            pages.extend(data.get("results"))

            if not data.get("has_more"):
                break

            query_dict["start_cursor"] = data.get("next_cursor")

        page_ids = [page["id"] for page in pages]
        print(f"Found {len(page_ids)} pages in Notion database {self.database_id}")

        return page_ids

    def load_page(self, page_id: str) -> List[Document]:
        """Read a page."""
        is_pdf = False
        data = self._request(PAGE_URL.format(page_id=page_id))

        # load properties as metadata
        metadata: Dict[str, Any] = {}

        for prop_name, prop_data in data["properties"].items():
            prop_type = prop_data["type"]

            if prop_type == "rich_text":
                value = (
                    prop_data["rich_text"][0]["plain_text"]
                    if prop_data["rich_text"]
                    else None
                )
            elif prop_type == "title":
                value = (
                    prop_data["title"][0]["plain_text"] if prop_data["title"] else None
                )
            elif prop_type == "multi_select":
                value = (
                    [item["name"] for item in prop_data["multi_select"]]
                    if prop_data["multi_select"]
                    else []
                )
            elif prop_type == "select":
                value = (
                    prop_data["select"]["name"] if prop_data["select"] else None
                )
            elif prop_type == "date":
                value = (
                    prop_data["date"]["start"] if prop_data["date"] else None
                )
            elif prop_type == "checkbox":
                value = (
                    prop_data["checkbox"]
                )
                if prop_name.lower() == "pdf" and value is True:
                    is_pdf = True
            elif prop_type == "url":
                value = (
                    prop_data["url"]
                )
            else:
                print(f"Unknown prop_type: {prop_type} for Notion page id: {page_id}")
                value = None

            metadata[prop_name.lower()] = value

        metadata["id"] = page_id
        page_content = self._load_blocks(block_id=page_id)

        """ validate """
        if not page_content:
            raise ValueError(f"No content found for page_id: '{page_id}', title: '{metadata['title']}'")
        if not metadata["source"]:
            raise ValueError(f"source: '{metadata['source']} not found for page_id: '{page_id}', title: '{metadata['title']}'")

        """ check status """
        if metadata["status"] in ["Archived", "Indexed"]:
            return []

        """ filter metadata """
        metadata_filtered = dict(filter(metadata_filter, metadata.items()))

        if is_pdf:
            print(f"\n\nLoading PDF '{metadata}'")
            docs = _get_pdf_content(page_content, page_id)
            return [Document(page_content=doc.page_content, metadata=metadata_filtered) for doc in docs]
        else:
            print(f"\n\nLoading Notion Page '{metadata}'")

        return [Document(page_content=page_content, metadata=metadata_filtered)]

    def _load_blocks(self, block_id: str, num_tabs: int = 0) -> str:
        """Read a block and its children."""
        result_lines_arr: List[str] = []
        cur_block_id: str = block_id

        while cur_block_id:
            data = self._request(BLOCK_URL.format(block_id=cur_block_id))

            for result in data["results"]:
                result_obj = result[result["type"]]

                if result["type"] == "file" or result["type"] == "pdf":
                    return result["file"]["file"]["url"]
                if "rich_text" not in result_obj:
                    continue

                cur_result_text_arr: List[str] = []

                for rich_text in result_obj["rich_text"]:
                    if "text" in rich_text:
                        cur_result_text_arr.append(
                            "\t" * num_tabs + rich_text["text"]["content"]
                        )

                if result["has_children"]:
                    children_text = self._load_blocks(
                        block_id=result["id"], num_tabs=num_tabs + 1
                    )
                    cur_result_text_arr.append(children_text)

                result_lines_arr.append("\n".join(cur_result_text_arr))

            cur_block_id = data.get("next_cursor")

        return "\n".join(result_lines_arr)

    def _request(
            self, url: str, method: str = "GET", query_dict: Dict[str, Any] = {}
    ) -> Any:
        """ Make a request to the Notion API.
        Include retry logic and rate limit handling. """
        # https://scrapeops.io/python-web-scraping-playbook/python-requests-retry-failed-requests/
        for _ in range(RETRY_COUNT):
            if WAIT is not None:
                time.sleep(WAIT)

            try:
                response = requests.request(
                    method,
                    url,
                    headers=self.headers,
                    json=query_dict,
                    timeout=TIMEOUT,
                )
                # response.raise_for_status()
                if response.status_code in [429, 500, 502, 503, 504]:
                    print(f"Got {response.status_code} from Notion API. Retrying...")
                    continue
                return response.json()
            except requests.exceptions.ConnectionError:
                pass
            return None
