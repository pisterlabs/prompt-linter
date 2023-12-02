import pickle
from typing import Optional

from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun,
                                         CallbackManagerForToolRun)
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import BaseTool
from langchain.vectorstores import FAISS


class AaveContextProvider(BaseTool):
    name = "aave_context_provider"
    description = """Aave expert who will tell you how to use the aave staking/lending protocol.
                    Uses: Credit Delegation,Flash Loans,Liquidations,Governance,Core Contracts,Periphery Contracts,Tokens,Deployed Contracts
                    You will provide this expert with a summary of what you want to do.
                    Returns text and/or javascript code."""

    def load_vector_store(self):
        """
        Loads the airstack vector store from pickle into a local file
        """
        global vector_store
        vector_store = FAISS.load_local("aave_faiss_index", OpenAIEmbeddings())

    def _run(self, query: str) -> str:
        """
        Search the docs and return the most relevant pages content which we will use to feed to the model
        """
        self.load_vector_store()
        res = vector_store.similarity_search(query)
        if res:
            return f"{res[0].page_content} {res[1].page_content} {res[2].page_content} ||| {res[0].metadata['source']},{res[1].metadata['source']},{res[2].metadata['source']}"

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("airstack_doc_search does not support async")
