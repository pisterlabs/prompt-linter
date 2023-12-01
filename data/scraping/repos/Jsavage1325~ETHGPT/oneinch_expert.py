from typing import Optional

from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun,
                                         CallbackManagerForToolRun)
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import BaseTool
from langchain.vectorstores import FAISS


class OneInchContextProvider(BaseTool):
    name = "one_inch_context_provider"
    description = """Expert who will answer specific questions related to 1inch. 1inch is decentrilised exchange aggregator (DEX). 
                    Uses: Aggregation Protocol (optimal rates, fast swaps) , Limit Order Protocol (flexibile order limits any EVM chains), FusionSwap (gasless front-run resistant swaps with variable exchange), 1inch network DAO.
                    Return text or javascript."""

    def load_vector_store(self):
        """
        Loads the langchain vector store from pickle into a local file
        """
        embeddings = OpenAIEmbeddings()
        global vector_store
        vector_store = FAISS.load_local("1inch_faiss_index", embeddings)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Search the 1inch docs and return the most relevant pages content which we will use to feed to the model
        """
        self.load_vector_store()
        res = vector_store.similarity_search(query)
        if res:
            return f"{res[0].page_content} {res[1].page_content} {res[2].page_content} ||| {res[0].metadata['source']},{res[1].metadata['source']},{res[2].metadata['source']}"
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("1inch_doc_search does not support async")
