from typing import List, Optional
from langchain.docstore.document import Document

from langchain.document_loaders.base import BaseLoader


class FaunaLoader(BaseLoader):
    def __init__(
            self,
            query: str,
            page_content_field: str,
            metadata_fields: Optional[List[str]] = None,
    ):
        self.query = query
        self.page_content_field = page_content_field
        self.metadata_fields = metadata_fields

    def load(self) -> List[Document]:
        try:
            from fauna import fql, Page
            from fauna.client import Client
            from fauna.encoding import QuerySuccess
        except ImportError:
            raise ValueError(
                "Could not import fauna python package. "
                "Please install it with `pip install fauna`."
            )

        with Client() as client:
            documents = []
            response: QuerySuccess = client.query(fql(self.query))
            page: Page = response.data
            while True:
                for result in page:
                    if result is not None:
                        document: Document = Document()
                        document.page_content = result[self.page_content_field]
                        document.metadata = dict()
                        for metadata_field in self.metadata_fields:
                            document.metadata[metadata_field] = result[metadata_field]
                        documents.append(document)
                if page.after is None:
                    break
                result: QuerySuccess = client.query()
                page: Page = result.data
            return documents
