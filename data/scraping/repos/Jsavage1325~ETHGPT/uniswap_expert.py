import pickle
from typing import Optional

from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun,
                                         CallbackManagerForToolRun)
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import BaseTool
from langchain.vectorstores import FAISS


class UniswapContextProvider(BaseTool):
    name = "uniswap_context_provider"
    description = """Uniswap expert who will tell you how to use the uniswap staking/lending protocol.
                    Uses:Uniswap Subgraph(indexing and organizing data),UniversalRouter(eth swap router),V3 Protocol,V3 SDK(simplifies interaction with Uniswap V3 smart contracts),Swap Widget(simplifies embedding the swapping experience into apps using one line of react code),web3-react(simplifies connecting dApps to various wallets through web3 connectors.),
                    You will provide this expert with a summary of what you want to do.
                    Returns text and/or javascript code."""

    def load_vector_store(self):
        """
        Loads the airstack vector store from pickle into a local file
        """
        global vector_store
        vector_store = FAISS.load_local("uniswap_v3_faiss_index", OpenAIEmbeddings())

    def _run(self, query: str) -> str:
        """
        Search the docs and return the most relevant pages content which we will use to feed to the model
        """
        self.load_vector_store()
        res = vector_store.similarity_search(query)
        # link to document (?)
        # res[0].metadata['source']
        if res:
            return f"{res[0].page_content} {res[1].page_content} {res[2].page_content} ||| {res[0].metadata['source']},{res[1].metadata['source']},{res[2].metadata['source']}"

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("airstack_doc_search does not support async")
