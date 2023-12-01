# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Module for MSTICPy natural language code search."""
import contextlib
import logging
from pathlib import Path
from typing import Optional

from IPython.display import Markdown

from langchain.document_loaders import GitLoader

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import VectorStore

from retrieval_base import RetrievalBase
from vector_stores import create_vector_store, read_vector_store
from _version import VERSION

__version__ = VERSION
__author__ = "Ian Hellen"


logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make
up an answer.

{context}

Question: {question}

When answering give preference to functionality that is exposed
via a pandas extension method (e.g. `df.mp_plot.timeline()`) or
a pivot function (e.g. `IpAddress.whois()`) over a standard function.

Supply any relevant code examples and include with your answer
(ensure you delimit the code with triple backticks and add the language name).

Return the answer in Markdown format.
"""

MP_VS_PATH = "E:/src/chat-langchain/mp-repo-vs.pkl"


class CodeSearch(RetrievalBase):
    """Ask questions about the code."""

    _PROMPT_TEMPLATE = PROMPT_TEMPLATE

    def __init__(
        self,
        vs_path: str = MP_VS_PATH,
        max_tokens: int = 1000,
        memory: bool = False,
        verbose: bool = False,
        model_name: str = "gpt-4",
    ):
        """Initialize the code search."""
        super().__init__(vs_path or MP_VS_PATH, max_tokens, memory, verbose, model_name=model_name)

    @classmethod
    def create_vectorstore(cls, input_path: str, vs_path: str):
        """
        Read in MSTICPy repo to vector store and save.

        Parameters
        ----------
        input_path : str
            Path to read MP repo from.
        vs_path : str
            Path to save pickled vectorstore to.

        """
        if Path(vs_path).exists():
            with contextlib.suppress(ValueError):
                return read_vector_store(vs_path, f"{cls.__name__}.create_vector_store")
        logger.info("Loading files from %s", input_path)
        print("Loading code files from", input_path, "...")
        loader = GitLoader(
            repo_path=input_path,
            file_filter=lambda file_path: (file_path.endswith(".py")),
        )
        return create_vector_store(loader, vs_path)

    def ask(self, question: str):
        """Ask a question about MSTICPy code and return the answer as Markdown."""
        return Markdown(self.retriever(question)["result"])

    # def create_qa_retriever(
    #     self,
    #     vectorstore: VectorStore,
    #     max_tokens: int = 2000,
    #     verbose: bool = False,
    #     memory: Optional[ConversationSummaryBufferMemory] = None,
    #     model_name: str = "gpt-4",
    # ):
    #     """Create a retriever."""
    #     prompt = PromptTemplate(
    #         template=self._PROMPT_TEMPLATE, input_variables=["context", "question"]
    #     )
    #     chain_type_kwargs = {"prompt": prompt}

    #     model = ChatOpenAI(model=model_name, temperature=0, max_tokens=max_tokens)
    #     return ConversationalRetrievalChain.from_llm(
    #         model,
    #         chain_type="stuff",
    #         retriever=vectorstore.as_retriever(),
    #         combine_docs_chain_kwargs={"prompt": prompt},
    #         # chain_type_kwargs=chain_type_kwargs,
    #         memory=memory,
    #         verbose=verbose,
    #     )
