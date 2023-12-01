"""Notion DB loader for langchain"""
import itertools
import time
from typing import Any, Dict, List
import requests
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from notion_utils.MyPyPDFLoader import MyPyPDFLoader

NOTION_BASE_URL = "https://api.notion.com/v1"
DATABASE_URL = NOTION_BASE_URL + "/databases/{database_id}/query"
PAGE_URL = NOTION_BASE_URL + "/pages/{page_id}"
BLOCK_URL = NOTION_BASE_URL + "/blocks/{block_id}/children"
QUERY_DICT = {
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
                    "equals": "Reviewed"
                }
            }
        ]
    },
    'page_size': 100
}


def _get_pdf_content(url_str: str,
                     page_id: str,
                     verbose: bool
                     ) -> List[Document]:
    if url_str.startswith("http"):
        loader = MyPyPDFLoader(url_str, verbose=verbose)
        pages = loader.load()
        return pages
    raise ValueError(f"Invalid URL of pdf: '{url_str}' at page_id: '{page_id}'")


def _read_metadata(page_id: str,
                   page_summary: Dict[str, Any],
                   metadata: Dict[str, Any]
                   ):
    """Reading metadata from Notion page summary."""
    for prop_name, prop_data in page_summary["properties"].items():
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
        elif prop_type == "unique_id":
            value = (
                prop_data["unique_id"]["number"] if prop_data["unique_id"] else None
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
        elif prop_type == "number":
            value = (
                prop_data["number"]
            )
        elif prop_type == "created_time":
            value = (
                prop_data["created_time"]
            )
        elif prop_type == "formula":
            value = (
                prop_data["formula"]
            )
        else:
            print(f"Unknown prop_type: {prop_type} for Notion page id: {page_id}")
            value = None

        metadata[prop_name.lower()] = value

    metadata["id"] = page_id


class MyNotionDBLoader(BaseLoader):
    """Notion DB Loader.
    Reads content from pages within a Noton Database.
    Args:
        integration_token (str): Notion integration token.
        database_id (str): Notion database id.
        verbose (bool): Whether to print debug messages.
        timeout (int): Timeout for requests to Notion API.
        wait (int): Wait time between retries.
        retry_count (int): Number of retries.
        metadata_filter_list (list[str]): List of metadata to keep.
        validate_missing_content (bool): Whether to validate missing content.
        validate_missing_metadata (list[str]): List of metadata to validate.
    """

    def __init__(self,
                 integration_token: str,
                 database_id: str,
                 verbose: bool,
                 timeout: int = 10000,
                 wait: int = 1,
                 retry_count: int = 5,
                 metadata_filter_list: list[str] = ['id', 'title'],
                 validate_missing_content: bool = True,
                 validate_missing_metadata: list[str] = ['source'],
                 ) -> None:
        """Initialize with parameters."""
        if not integration_token:
            raise ValueError("integration_token must be provided")
        if not database_id:
            raise ValueError("database_id must be provided")

        self.token = integration_token
        self.database_id = database_id
        self.verbose = verbose
        self.headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }
        self.timeout = timeout
        self.wait = wait
        self.retry_count = retry_count
        self.metadata_filter_list = metadata_filter_list
        self.validate_missing_content = validate_missing_content
        self.validate_missing_metadata = validate_missing_metadata

    def load(self,
             query_dict: Dict[str, Any] = QUERY_DICT
             ) -> List[Document]:
        """Load documents from the Notion database.
        Args:
            query_dict (Dict[str, Any]): Query dict for Notion API.
        Returns:
            List[Document]: List of documents.
        """
        page_summaries = self._retrieve_page_summaries(query_dict)
        print(f"Found {len(page_summaries)} pages in Notion database {self.database_id}\n")
        return list(itertools.chain.from_iterable(self.load_page(page_summary) for page_summary in page_summaries))

    def _retrieve_page_summaries(
            self,
            query_dict: Dict[str, Any] = QUERY_DICT
    ) -> List[Dict[str, Any]]:
        """Get all the pages from a Notion database."""
        pages: List[Dict[str, Any]] = []

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

        return pages

    # def _retrieve_page_ids(
    #         self,
    #         query_dict: Dict[str, Any] = QUERY_DICT
    # ) -> List[str]:
    #     """Get page ids"""
    #     pages = self._retrieve_pages(query_dict)
    #
    #     page_ids = [page["id"] for page in pages]
    #     print(f"Found {len(page_ids)} pages in Notion database {self.database_id}\n")
    #
    #     return page_ids

    def duplicates(
            self,
            query_dict: Dict[str, Any] = {}
    ) -> List[tuple[str, str]]:
        """Get duplicate pages"""
        pages = self._retrieve_page_summaries(query_dict)

        list_items = [(page["id"], page["properties"]["id"]["title"][0]["plain_text"]) for page in pages]
        from collections import Counter
        second_items = [t[1] for t in list_items]
        duplicates = [item for item, count in Counter(second_items).items() if count > 1]
        duplicate_tuples = [t for t in list_items if t[1] in duplicates]
        return duplicate_tuples

    def load_page(self,
                  page_summary: Dict[str, Any]
                  ) -> List[Document]:
        """Read a page."""
        is_pdf = False
        page_id = page_summary["id"]

        # load properties as metadata
        metadata: Dict[str, Any] = {}

        """ extract metadata """
        _read_metadata(page_id, page_summary, metadata)

        """ load all blocks of content """
        page_content = self._load_blocks(block_id=page_id)

        """ validate """
        if not page_content and self.validate_missing_content:
            raise ValueError(f"No content found for page_id: '{page_id}', metadata: '{metadata}'")
        if self.validate_missing_metadata:
            for missing_metadata in self.validate_missing_metadata:
                if missing_metadata not in metadata:
                    raise ValueError(
                        f"Missing metadata: '{missing_metadata}' for page_id: '{page_id}', metadata: '{metadata}'")

        """ check status """
        if 'status' in metadata and metadata["status"] in ["Archived", "Indexed"]:
            return []

        """ filter metadata """
        metadata_filtered = {k: v for k, v in metadata.items() if any(x == k for x in self.metadata_filter_list)}

        if is_pdf:
            """ get notion url to pdf and extract content """
            print(f"Loading PDF '{metadata}'\n")
            print(f"page_content: {page_content}\n") if self.verbose else None
            docs = _get_pdf_content(page_content, page_id, verbose=self.verbose)
            return [Document(page_content=doc.page_content, metadata=metadata_filtered) for doc in docs]
        else:
            print(f"Loading Notion Page '{metadata_filtered}'\n")

        return [Document(page_content=page_content, metadata=metadata_filtered)]

    def _load_blocks(self,
                     block_id: str,
                     num_tabs: int = 0
                     ) -> str:
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
            self,
            url: str,
            method: str = "GET",
            query_dict: Dict[str, Any] = {}
    ) -> Any:
        """ Make a request to the Notion API.
        Include retry logic and rate limit handling. """
        # https://scrapeops.io/python-web-scraping-playbook/python-requests-retry-failed-requests/
        for i in range(self.retry_count):
            time.sleep(i * self.wait + 1)

            try:
                response = requests.request(
                    method,
                    url,
                    headers=self.headers,
                    json=query_dict,
                    timeout=self.timeout,
                )
                # response.raise_for_status()
                if response.status_code in [429, 500, 502, 503, 504]:
                    print(f"Got {response.status_code} from Notion API. Retrying...")
                    continue
                return response.json()
            # except requests.exceptions.ConnectionError:
            except:
                continue
        raise ValueError(f"Failed to get response from Notion API after {self.retry_count} retries")
