from typing import Optional

from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun,
                                         CallbackManagerForToolRun)
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import BaseTool
from langchain.vectorstores import FAISS


class GnosisContextProvider(BaseTool):
    name = "gnosis_context_provider"
    description = """Expert who will answer specific questions related to gnosis. Gnosis is and affordable Ethereum side chain. 
                    Uses:Deploying Smart Contracts,Interacting with Gnosis,Building Dapps,Verify Contracts,Bridge Tutorials,JSON RPC API Providers,Wallets,Faucets,Data&Analytics,Oracles,Beacon Chain
                    Return text or javascript."""

    def load_vector_store(self):
        """
        Loads the langchain vector store from pickle into a local file
        """
        embeddings = OpenAIEmbeddings()
        global vector_store
        vector_store = FAISS.load_local("gnosis_faiss_index", embeddings)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Search the gnosis docs and return the most relevant pages content which we will use to feed to the model
        """
        self.load_vector_store()
        res = vector_store.similarity_search(query)
        if res:
            # sauce res[0].metadata['source']
            return f"{res[0].page_content} {res[1].page_content} {res[2].page_content} ||| {res[0].metadata['source']},{res[1].metadata['source']},{res[2].metadata['source']}"
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("1inch_doc_search does not support async")
