from langchain.chains.base import Chain
from typing import Dict, List
from langchain.agents import Tool
# from paperai import paper_chat
from paperai.config import *
from qdrant_client import QdrantClient, models
from langchain.vectorstores import Qdrant
from paperai.vectordb import DatabaseInterface

class CSPaperChain(Chain, DatabaseInterface):
    chain: Chain
    output_key: str = "output"
    vectordb: DatabaseInterface

    @property
    def input_keys(self) -> List[str]:
        return list(self.chain.input_keys)

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        # Queries the database to get the relevant documents for a given query
        query = inputs.get("input_documents", "")
        found_docs = self.vectordb.qdrant_db.similarity_search(query, include_metadata=True)
        output = self.chain.run(input_documents=found_docs, question=query)
        if found_docs and len(found_docs) > 0 and found_docs[0].metadata:
            document = found_docs[0]
            metadata = document.metadata
            source = metadata.get("source", None)
            page = metadata.get("page", -1)
            output += f"***%Page Content: {document.page_content}"
            output += f"***%Source: {source}"
            output += f"***%Page: {page}"
        return { self.output_key : output }