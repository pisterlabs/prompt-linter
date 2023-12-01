"""
    The following code is under CC-BY-NC-SA 4.0 license (more in root/LICENSE.txt)
            Free to use and redistribute for any non-commercial purpose
"""

from typing import Any, List, Optional, Type
from pydantic import BaseModel, Extra, Field
from langchain.base_language import BaseLanguageModel
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma


def _get_default_text_splitter() -> TextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


class VectorStoreIndexWrapper(BaseModel):
    """Wrapper around a vectorstore for easy access."""

    vectorstore: VectorStore

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def query(
            self, question: str, llm: Optional[BaseLanguageModel] = None, **kwargs: Any
    ) -> str:
        """Query the vectorstore."""
        llm = llm or OpenAI(temperature=0)
        chain = RetrievalQA.from_chain_type(
            llm, retriever=self.vectorstore.as_retriever(), **kwargs
        )
        return chain.run(query=question)

    def query_with_sources(
            self, question: str, llm: Optional[BaseLanguageModel] = None, **kwargs: Any
    ) -> dict:
        """Query the vectorstore and get back sources."""
        llm = llm or OpenAI(temperature=0)
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm, retriever=self.vectorstore.as_retriever(), **kwargs
        )
        return chain({chain.question_key: question})


class VectorstoreIndexCreator(BaseModel):
    """Logic for creating indexes."""

    vectorstore_cls: Type[VectorStore] = Chroma
    embedding: Embeddings = Field(default_factory=OpenAIEmbeddings)
    text_splitter: TextSplitter = Field(default_factory=_get_default_text_splitter)
    vectorstore_kwargs: dict = Field(default_factory=dict)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def from_loaders(self, loaders: List[BaseLoader]) -> VectorStoreIndexWrapper:
        """Create a vectorstore index from loaders."""
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        return self.from_documents(docs)

    def from_documents(self, documents: List[Document]) -> VectorStoreIndexWrapper:
        """Create a vectorstore index from documents."""
        sub_docs = self.text_splitter.split_documents(documents)
        vectorstore = self.vectorstore_cls.from_documents(
            sub_docs, self.embedding, **self.vectorstore_kwargs
        )
        return VectorStoreIndexWrapper(vectorstore=vectorstore)

    def from_persistent_index(self, path: str) -> VectorStoreIndexWrapper:
        """Load a vectorstore index from a persistent index."""
        vectorstore = self.vectorstore_cls(persist_directory=path, embedding_function=self.embedding)
        return VectorStoreIndexWrapper(vectorstore=vectorstore)
