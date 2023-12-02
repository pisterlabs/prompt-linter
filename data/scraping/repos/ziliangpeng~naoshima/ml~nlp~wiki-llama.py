import os
import pickle
from llama_index import SimpleDirectoryReader, Document, VectorStoreIndex, TreeIndex
from loguru import logger

from llama_index import ServiceContext
from llama_index.embeddings import OpenAIEmbedding

from llama_index.llms.base import LLM
from llama_index.llms.utils import LLMType, resolve_llm
from llama_index.llms.openai import OpenAI
from llama_index.llms.mock import MockLLM

from typing import (
    Any,
    Sequence,
)

from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
)


def load_documents():
    naoshima_dir = os.environ["NAOSHIMA"]
    wiki_pages_url = os.path.join(naoshima_dir, "crawler/wiki-pages/en")
    documents = SimpleDirectoryReader(wiki_pages_url).load_data()
    logger.info(f"Loaded {len(documents)} documents")
    return documents


class O:
    pass


class ZLLM(OpenAI):
    # @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        logger.info("Chatting...")
        # logger.info(messages)
        logger.info(kwargs)
        resp = O()
        resp.message = O()
        resp.message.content = "Hello"
        return resp


def main():
    logger.info("Starting...")
    cached_openai_key = os.environ.get("OPENAI_API_KEY")
    # invalidate the key so that the service context will use the local model
    os.environ["OPENAI_API_KEY"] = "false"

    docs = load_documents()

    if False and os.path.isfile("wiki-llama.pkl"):
        new_index = pickle.load(open("wiki-llama.pkl", "rb"))
        logger.info("Index loaded")
    else:
        # Can use either MockLLM or ZLLM
        llm = MockLLM()
        llm = ZLLM()
        service_context = ServiceContext.from_defaults(embed_model="local", llm=llm)
        # idx = VectorStoreIndex
        idx = TreeIndex
        new_index = idx.from_documents(docs, service_context=service_context)
        logger.info("Index created")
        # pickle.dump(new_index, open("wiki-llama.pkl", "wb"))
        logger.info("Index saved")

    # set Logging to DEBUG for more detailed outputs
    query_engine = new_index.as_query_engine(service_context=service_context)
    # print(cached_openai_key)
    # os.environ["OPENAI_API_KEY"] = cached_openai_key
    while True:
        # query = input("Enter your question: ")
        query = "What is Python?"
        if query == "exit":
            break
        response = query_engine.query(query)
        print(response)
        break


if __name__ == "__main__":
    main()
