"""Tool for the SRD API's /classes endpoint."""

import json
from typing import Optional
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun)
from langchain.tools.base import BaseTool

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class dndSRD(BaseTool):
    """Tool that adds the capability to search D&D 5E SRD."""

    name = "DNDSRD"
    db: Chroma = None
    description = (
    "Used when DND5E gives a 404 error to retrieve information about the rules and content of Dungeons and Dragons"
    "The Action Input should be a brief query for the in-context subject matter you want to learn about. "
    "Retrieved information is deterministic, so the same Action Input will always yield the same output, not new information"
    # "For example, you can use an Action Input of \"Bard hit dice\" where Bard is the context and hit dice is the subject matter, this will retrieve an excerpt from the rules book on the Bard class's hit dice value"
    )

    def __init__(self):
        super().__init__()
        embeddings = HuggingFaceEmbeddings()
        persist_directory = '.chromaDB/srd'
        chromaDb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        self.db = chromaDb

    def _run(self, ruleBookQuery: str) -> str:
        print("")
        print(f"==== DNDSRD qry: `{ruleBookQuery}`")
        results = self.db.similarity_search(ruleBookQuery, k=6)
        output = f"""DND Search Results:"""
        seen = set()
        for index, doc in enumerate(results):
            if len(seen) >= 2:
                break
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                output += f"""
[rank {index+1}] {doc.page_content}"""

        return output

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the DNDSRD tool asynchronously."""
        raise NotImplementedError("DNDSRD does not support async")