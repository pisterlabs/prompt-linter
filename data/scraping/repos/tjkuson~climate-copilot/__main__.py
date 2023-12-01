"""Entry point for the application."""
from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

import pinecone
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from climate_copilot.load import load_resources
from climate_copilot.utils import pinecone_environment

if TYPE_CHECKING:
    from climate_copilot.utils import PineconeEnvironment


def ask(query: str, pinecone_env: PineconeEnvironment) -> str | None:
    """Ask a question to the chatbot."""
    pinecone.init(api_key=pinecone_env.api_key, environment=pinecone_env.environment)
    embeddings = OpenAIEmbeddings()  # type: ignore[call-arg]
    doc_search = Pinecone.from_existing_index(
        index_name=pinecone_env.index_name,
        embedding=embeddings,
    )
    chat = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=1,
        verbose=True,
    )  # type: ignore[call-arg]
    qa = RetrievalQA.from_chain_type(
        chain_type="stuff",
        llm=chat,
        retriever=doc_search.as_retriever(),
    )
    return qa({"query": query}).get("result")


def converse(pinecone_env: PineconeEnvironment) -> None:
    """Converse with the chatbot."""
    pinecone.init(api_key=pinecone_env.api_key, environment=pinecone_env.environment)
    embeddings = OpenAIEmbeddings()  # type: ignore[call-arg]
    doc_search = Pinecone.from_existing_index(
        index_name=pinecone_env.index_name,
        embedding=embeddings,
    )
    chat = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=1,
        verbose=True,
    )  # type: ignore[call-arg]
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=doc_search.as_retriever(),
    )
    chat_history: list[tuple[str, str]] = []
    print("Start a conversation with Climate Copilot! (type 'quit' to exit)")
    while True:
        query = input("You: ")
        if query == "quit":
            break
        response = qa({"question": query, "chat_history": chat_history}).get("answer")
        if response is None:
            print(
                "Could not retrieve an answer from Climate Copilot. Try again.",
                file=sys.stderr,
            )
            continue
        print(f"Climate Copilot: {response}")
        chat_history.append((query, response))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="climate-copilot",
        description="Chatbot that answers questions about climate change.",
    )
    parser.add_argument(
        "--load-resources",
        action="store_true",
        help="Load the resources into Pinecone.",
    )
    parser.add_argument(
        "--ask",
        type=str,
        help="Ask a single question to the chatbot.",
    )
    parser.add_argument(
        "--converse",
        action="store_true",
        help="Converse with the chatbot.",
    )
    args = parser.parse_args()

    pinecone_env = pinecone_environment()
    if args.load_resources:
        load_resources(pinecone_env)
    if args.ask:
        print(ask(args.ask, pinecone_env))
    elif args.converse:
        converse(pinecone_env)
