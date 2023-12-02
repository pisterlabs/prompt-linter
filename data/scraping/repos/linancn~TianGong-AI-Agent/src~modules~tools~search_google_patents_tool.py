from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.document_loaders import BigQueryLoader
from langchain.schema.document import Document
from langchain.tools import BaseTool
from pydantic import BaseModel


class SearchGooglePatentsTool(BaseTool):
    name = "search_google_patents"
    description = "Search patents through google big query."

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema

    def create_query(self, search_term: str, return_count: int = 16) -> list[Document]:
        q = f"""
  WITH 
  pubs as (
    SELECT DISTINCT 
      pub.publication_number
    FROM `patents-public-data.patents.publications` pub
      INNER JOIN `patents-public-data.google_patents_research.publications` gpr ON
        pub.publication_number = gpr.publication_number
    WHERE 
      "{search_term}" IN UNNEST(gpr.top_terms)
  )

  SELECT
    publication_number, title, abstract, url, country
  FROM 
    `patents-public-data.google_patents_research.publications`
  WHERE
    publication_number in (SELECT publication_number from pubs)
    LIMIT {return_count}
"""
        return q

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously."""
        BASE_QUERY = self.create_query(search_term=query, return_count=16)

        loader = BigQueryLoader(
            BASE_QUERY,
            page_content_columns=["abstract"],
            metadata_columns=["publication_number", "title", "country", "url"],
        )
        docs = loader.load()

        docs_list = []
        for doc in docs:
            source_entry = "[{}. {}. {}.]({})".format(
                doc.metadata["publication_number"],
                doc.metadata["title"],
                doc.metadata["country"],
                doc.metadata["url"],
            )
            docs_list.append({"content": doc.page_content, "source": source_entry})

        return docs_list

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        BASE_QUERY = self.create_query(search_term=query, return_count=16)

        loader = BigQueryLoader(
            BASE_QUERY,
            page_content_columns=["abstract"],
            metadata_columns=["publication_number", "title", "country", "url"],
        )
        docs = loader.load()

        docs_list = []
        for doc in docs:
            source_entry = "[{}. {}. {}.]({})".format(
                doc.metadata["publication_number"],
                doc.metadata["title"],
                doc.metadata["country"],
                doc.metadata["url"],
            )
            docs_list.append({"content": doc.page_content, "source": source_entry})

        return docs_list
