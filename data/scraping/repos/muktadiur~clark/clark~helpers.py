import os
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain
)
from clark.base import BaseConversation
from clark.gpt4all import GPT4AllConversation
from clark.openai import OpenAIConversation
from clark.hf import HFConversation
from clark.document import process_documents


def get_converstation() -> BaseConversation:
    if os.getenv("CONVERSATION_ENGINE") == "gpt4all":
        return GPT4AllConversation()

    if os.getenv("CONVERSATION_ENGINE") == "hf":
        return HFConversation()

    return OpenAIConversation()


def get_chain() -> BaseConversationalRetrievalChain:
    conversation = get_converstation()
    texts = process_documents()
    return conversation.get_chain(texts=texts)
