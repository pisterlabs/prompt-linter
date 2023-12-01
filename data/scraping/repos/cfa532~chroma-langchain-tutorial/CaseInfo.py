"""Wrapper around wikipedia API."""

from typing import Union
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
# from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from config import CHROMA_CLIENT, EMBEDDING_FUNC

class CaseInfo(Docstore):
    """Query lawsuit docments to find basic information"""

    def __init__(self, collection_name: str) -> None:
        """
        case_info is of type LegalCase, whose id is used as collection_name for the case in Chroma. 
        basic information about the case, including name of plaintiff, defendant, judge, lawyer
        and briefing, collection_name create for the case documents.
        """
        self._db = Chroma(collection_name=collection_name, client=CHROMA_CLIENT, embedding_function=EMBEDDING_FUNC)

        """Check that wikipedia package is installed."""
        """search function for baidu, 天眼查 can be implemented
        try:
            import wikipedia  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import wikipedia python package. "
                "Please install it with `pip install wikipedia`."
            )
        """

    def search(self, search: str) -> Union[str, Document]:
        """Try to search for wiki page.

        If page exists, return the page summary, and a PageWithLookups object.
        If page does not exist, return similar entries.

        Args:
            search: search string.

        Returns: a Document object or error message.
        """
        try:
            docs = self._db.similarity_search(search)
            page_content = docs[0].page_content
            result: Union[str, Document] = Document(
                page_content=page_content, metadata=docs[0].metadata
            )
        except Exception:
            result = f"Could not find [{search}]."
        return result
