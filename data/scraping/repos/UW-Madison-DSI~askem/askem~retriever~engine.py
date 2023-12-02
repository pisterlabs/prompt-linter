import logging
import os
from typing import Iterator

import langchain
import requests
import tenacity
from base import get_client, get_documents
from data_models import Document
from langchain.agents import initialize_agent
from langchain.agents.agent_iterator import AgentExecutorIterator

WEAVIATE_CLIENT = get_client()


def query_xdd(query: str, top_k: int, dataset: str) -> dict:
    """Query xdd articles API.

    Args:
        query: Query string.
        top_k: Number of documents to return. e.g.: 5.
        dataset: Dataset to query. e.g.: covid.
    """

    url = os.getenv("HYBRID_SEARCH_XDD_URL")
    logging.debug(f"Accessing XDD elastic search at {url}")

    params = {
        "term": query,
        "dataset": dataset,
        "max": top_k,
        "match": "true",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_contents(response: dict, path: list, field: str) -> list[any]:
    """Get list of _gddid values from response."""

    for key in path:
        response = response[key]
    return [hit[field] for hit in response]


# ========== Internal access to all endpoints ==========


# Vector search
def vector_search(**kwargs) -> list[Document]:
    return get_documents(client=WEAVIATE_CLIENT, **kwargs)


# Hybrid search
def hybrid_search(
    question: str,
    topic: str,
    screening_top_k: int = 100,
    **kwargs,
) -> list[Document]:
    # Screening paper by xdd ElasticSearch
    results = query_xdd(question, screening_top_k, dataset=topic)
    paper_ids = get_contents(results, ["success", "data"], "_gddid")

    return get_documents(
        question=question,
        topic=topic,
        client=WEAVIATE_CLIENT,
        paper_ids=paper_ids,
        **kwargs,
    )


# React search
@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=3, max=15),
    stop=tenacity.stop_after_attempt(6),
)
def get_llm(model_name: str):
    """Get LLM instance."""
    return langchain.chat_models.ChatOpenAI(model_name=model_name, temperature=0)


class ReactManager:
    """Manage information in a single search chain."""

    def __init__(
        self,
        entry_query: str,
        search_config: dict,
        openai_model_name: str,
        verbose: bool = False,
    ):
        self.entry_query = entry_query
        self.search_config = search_config
        self.openai_model_name = openai_model_name
        self.used_docs = []
        self.latest_used_docs = []

        # Retriever + ReAct agent
        self.agent_executor = initialize_agent(
            tools=self.react_tools(),
            llm=get_llm(self.openai_model_name),
            agent=langchain.agents.AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            handle_parsing_errors=True,
        )

    def react_tools(self) -> list[any]:
        """Tool set provided to ReAct."""
        hybrid_search = langchain.tools.StructuredTool.from_function(
            self._search_retriever
        )
        return [hybrid_search]

    def _search_retriever(self, question: str) -> str:
        """Useful when you need to answer question about facts."""
        # Do NOT change the doc-string of this function, it will affect how ReAct works!

        relevant_docs = hybrid_search(
            question=question,
            **self.search_config,
        )

        # Collect used documents
        self.used_docs.extend(relevant_docs)
        self.latest_used_docs = relevant_docs
        return "\n\n".join([r.text for r in relevant_docs])

    def get_iterator(self) -> AgentExecutorIterator:
        """ReAct iterator."""
        return self.agent_executor.iter(inputs={"input": self.entry_query})

    def run(self) -> str:
        """Run the chain until the end."""
        return self.agent_executor.invoke({"input", self.entry_query})["output"]


def react_search(
    question: str,
    openai_model_name: str = "gpt-4-1106-preview",
    streaming: bool = False,
    **kwargs,
) -> dict | Iterator[dict]:
    chain = ReactManager(
        entry_query=question,
        openai_model_name=openai_model_name,
        search_config=kwargs,
        verbose=False,
    )

    if not streaming:
        answer = chain.run()
        return {"answer": answer, "used_docs": chain.used_docs}

    return chain.get_iterator()
