from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
)
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import openai
import json


class DittoExampleStore:
    def __init__(self):
        # example template will be used to match query / example pairs.
        self.example_template = PromptTemplate(
            input_variables=["query", "answer"],
            template="Query: {query}\nAnswer: {answer}",
        )
        self.construct()

    def construct(self):
        self.create_query_example_pairs()
        self.create_example_store()

    def create_query_example_pairs(self):
        """creates query / example pairs and saves to self.example_pairs"""

        self.example_pairs = [
            {
                "query": "What are gas prices looking like today?",
                "answer": "<GOOGLE_SEARCH> gas prices today",
            },
            {
                "query": "What is the weather like in New York?",
                "answer": "<GOOGLE_SEARCH> weather in New York",
            },
            {
                "query": "Can you look up topic X or topic Y for me?",
                "answer": "<GOOGLE_SEARCH> topic X or topic Y",
            },
            {
                "query": "What is the cheapest flight to New York from San Francisco?",
                "answer": "<GOOGLE_SEARCH> cheapest flight from San Francisco to New York",
            },
            {
                "query": "Latest news on topic X or topic Y?",
                "answer": "<GOOGLE_SEARCH> latest news on topic X",
            },
            {
                "query": "What's the weather in Atlanta, GA?",
                "answer": "<GOOGLE_SEARCH> weather in Atlanta, GA",
            },
            {
                "query": "What is the current population of Tokyo, Japan?",
                "answer": "<GOOGLE_SEARCH> population of Recife, Brazil",
            },
            {
                "query": "What is the forecast for the next 5 days in Miami, FL?",
                "answer": "<GOOGLE_SEARCH> forecast for the next 5 days in Miami, FL",
            },
            {
                "query": "What is the current time in New York?",
                "answer": "<GOOGLE_SEARCH> current time in New York",
            },
            {
                "query": "Can you google search topic X or topic Y for me?",
                "answer": "<GOOGLE_SEARCH> topic X or topic Y",
            },
            {
                "query": "Google search the latest news on topic X or topic Y for me?",
                "answer": "<GOOGLE_SEARCH> latest news on topic X",
            },
            {
                "query": "Can you try looking up the weather in Maimi FL using Google?",
                "answer": "<GOOGLE_SEARCH> weather in Miami, FL",
            },
        ]

    def create_example_store(self):
        embeddings = OpenAIEmbeddings()
        self.example_store = MaxMarginalRelevanceExampleSelector.from_examples(
            # This is the list of examples available to select from.
            self.example_pairs,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            embeddings,
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # This is the number of examples to produce.
            k=5,
        )
        self.mmr_prompt = FewShotPromptTemplate(
            # We provide an ExampleSelector instead of examples.
            example_selector=self.example_store,
            example_prompt=self.example_template,
            prefix="Below are some examples of how how to use the tools:",
            suffix=" {query}",
            input_variables=["query"],
        )

    def get_example(self, query: str):
        example = str(self.mmr_prompt.format(query=query)).replace(query, "")
        return example


if __name__ == "__main__":
    ditto_example_store = DittoExampleStore()
