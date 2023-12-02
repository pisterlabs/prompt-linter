#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Literal, Optional

# chains
from langchain.chains import RetrievalQA

# embeddings
from langchain.embeddings import (
    HuggingFaceBgeEmbeddings
)

# llm
from langchain.llms import Ollama

# retrievers
from langchain.retrievers import SVMRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

# text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# vectorstores
from langchain.vectorstores import Chroma, Qdrant

from src.helper import log_method_call
from src.urlreader import Scraper


class llama_summarizer:
    """Summarize text from a URL.

    Attributes:
        url (str): URL to scrape.
        question (str): Question to ask the model.
        model (str): Model to use for summarization.
        base_url (str): Base URL for Ollama.
        verbose (bool): If True, print out debug information.
        chunk_size (int): Size of chunks to split text into.
        embedding_model (str): Embedding model to use.
        retriever (str): Retriever to use.
        device (str): Device to use.
        text (str): Scraped text from URL.
        splits (List[str]): Split text into chunks.
        vectorstore (VectorStore): Vector store for embeddings.
        llm (Ollama): Ollama language model.
        embeddings (Embeddings): Embeddings instance.
        qa_chain (RetrievalQA): QA chain for retrieval.
        answ (str): Generated answer.

    To summarize the text use the methods in the following order:
        scrape_text: Scrape text from URL.
        split_text: Split text into chunks.
        instantiate_embeddings: Instantiate embeddings.
        instantiate_llm: Instantiate Ollama.
        instantiate_retriever: Instantiate retriever.
        instantiate_qa_chain: Instantiate QA chain.
        generate: Generate answer.
    """

    def __init__(
        self,
        url: str,
        question: Optional[str] = "Summarize this text",
        model: Optional[str] = "summarizev2",
        base_url: Optional[str] = "http://localhost:11434",
        verbose: Optional[bool] = False,
        chunk_size: Optional[int] = 512,
        embedding_model: Literal["large", "small"] = "large",
        retriever: Literal["default", "SVM", "MultiQuery"] = "default",
        device: Literal["cpu", "mps", "cuda"] = "cpu",
    ):
        """Initialize summarizer with given parameters."""
        self.url = url
        self.question = question
        self.model = model
        self.base_url = base_url
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.embedding_model = embedding_model
        self.retriever = retriever
        self.device = device
        self.text = None
        self.splits = None
        self.vectorstore = None
        self.llm = None
        self.embeddings = None
        self.qa_chain = None
        self.answ = None

        if self.verbose:
            logging.basicConfig(
                filename="summarizer.log", level=logging.INFO
            )
            logging.info("\n \nInitializing summarizer")

    @log_method_call
    def scrape_text(self):
        """Scrape text from the provided URL using the Scraper class."""
        # assert that URL has been provided
        assert self.url is not None, "URL has not been provided"

        # Read text from URL
        scrpr = Scraper(url=self.url, first_three=False)

        self.text = f"[question: {self.question}] [text: {scrpr.scrape_text()}]"

        return self

    @log_method_call
    def split_text(self):
        """Split the scraped text into chunks using RecursiveCharacterTextSplitter."""
        # assert that text has been scraped
        assert (
            self.text is not None
        ), "Text has not been scraped, use the method scrape_text()"

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=0
        )
        self.splits = text_splitter.create_documents([self.text])
        return self

    @log_method_call
    def instantiate_embeddings(self):
        """Instantiate embeddings using either HuggingFaceBgeEmbeddings or SentenceTransformerEmbeddings."""
        if self.embedding_model == "large":
            bge_large = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-large-en",
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True},
            )
            self.embeddings = bge_large
        elif self.embedding_model == "small":
            bge_small = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-small-en",
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True},
            )
            self.embeddings = bge_small
        return self

    @log_method_call
    def instantiate_llm(self):
        """Instantiate the Ollama language model."""
        self.llm = Ollama(base_url=self.base_url, model=self.model)
        return self

    @log_method_call
    def instantiate_retriever(self):
        """Instantiate the retriever using either Chroma, SVMRetriever, or MultiQueryRetriever."""
        # assert that all necessary components have been instantiated
        assert (
            self.splits is not None
        ), "Text has not been split, use the method split_text()"
        assert (
            self.embeddings is not None
        ), "Embeddings have not been instantiated, use the method instantiate_embeddings()"
        assert (
            self.llm is not None
        ), "Ollama has not been instantiated, use the method instantiate_llm()"

        # Instantiate retriever
        if self.retriever == "default":
            self.vectorstore = Chroma.from_documents(
                self.splits, embedding=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever()
        elif self.retriever == "SVM":
            self.retriever = SVMRetriever.from_documents(
                self.splits, embeddings=self.embeddings
            )
        elif self.retriever == "MultiQuery":
            self.vectorstore = Qdrant.from_documents(
                self.splits, embedding=self.embeddings, location=":memory:"
            )
            self.retriever = MultiQueryRetriever.from_llm(
                retriever=self.vectorstore.as_retriever(), llm=self.llm
            )
        return self

    @log_method_call
    def instantiate_qa_chain(self):
        """Instantiate the QA chain using the RetrievalQA class."""
        # assert that all necessary components have been instantiated
        assert (
            self.llm is not None
        ), "Ollama has not been instantiated, use the method instantiate_llm()"
        assert (
            self.retriever is not None
        ), "Retriever has not been instantiated, use the method instantiate_retriever()"

        # Instantiate QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            self.llm, retriever=self.retriever
        )
        return self

    @log_method_call
    def generate(self):
        """Generate the answer using the instantiated QA chain."""
        # Lazy instantiation of necessary components
        if self.qa_chain is None:
            self.scrape_text()
            self.split_text()
            self.instantiate_embeddings()
            self.instantiate_llm()
            self.instantiate_retriever()
            self.instantiate_qa_chain()

        # Generate answer
        self.answ = self.qa_chain({"query": self.question})
        return self

    def __repr__(self):
        """Return a representation of the summarizer."""
        return f"Summarizer(url={self.url},\nquestion={self.question},\nmodel={self.model},\nbase_url={self.base_url},\nverbose={self.verbose},\nchunk_size={self.chunk_size},\nembedding_model={self.embedding_model},\nretriever={self.retriever},\ndevice={self.device})"

    def clear_cache(self):
        """Clear the cache."""
        self.text = None
        self.splits = None
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.answ = None
        self.device = None
        del self
