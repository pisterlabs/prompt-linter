import hnswlib
import os
import cohere
import json
import uuid
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from document import Documents

api_key = "fkAeCp5ZzmMiI4YBtkKUD6BanZVdk1vImBGZ5W0m"
co = cohere.Client(api_key)


class Chatbot:
    """
    A class representing a chatbot.

    Parameters:
    docs (Documents): An instance of the Documents class representing the collection of documents.

    Attributes:
    conversation_id (str): The unique ID for the conversation.
    docs (Documents): An instance of the Documents class representing the collection of documents.

    Methods:
    generate_response(message): Generates a response to the user's message.
    retrieve_docs(response): Retrieves documents based on the search queries in the response.

    """

    def __init__(self, docs: Documents):
        self.docs = docs
        self.conversation_id = str(uuid.uuid4())

    def generate_response(self, message: str):
        """
        Generates a response to the user's message.

        Parameters:
        message (str): The user's message.

        Yields:
        Event: A response event generated by the chatbot.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved documents.

        """
        # Generate search queries (if any)
        response = co.chat(
            message=message, model="command-nightly", search_queries_only=True
        )

        # If there are search queries, retrieve documents and respond
        if response.search_queries:
            print("Retrieving information...")

            documents = self.retrieve_docs(response)

            response = co.chat(
                # model=
                message=message,
                model="command-nightly",
                documents=documents,
                conversation_id=self.conversation_id,
                stream=True,
            )
            for event in response:
                yield event

        # If there is no search query, directly respond
        else:
            response = co.chat(
                message=message,
                model="command-nightly",
                conversation_id=self.conversation_id,
                stream=True,
            )
            for event in response:
                yield event

    def retrieve_docs(self, response) -> List[Dict[str, str]]:
        """
        Retrieves documents based on the search queries in the response.

        Parameters:
        response: The response object containing search queries.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved documents.

        """
        # Get the query(s)
        queries = []
        for search_query in response.search_queries:
            queries.append(search_query["text"])

        # Retrieve documents for each query
        retrieved_docs = []
        for query in queries:
            retrieved_docs.extend(self.docs.retrieve(query))

        # # Uncomment this code block to display the chatbot's retrieved documents
        # print("DOCUMENTS RETRIEVED:")
        # for idx, doc in enumerate(retrieved_docs):
        #     print(f"doc_{idx}: {doc}")
        # print("\n")

        return retrieved_docs
