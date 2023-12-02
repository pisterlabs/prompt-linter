from langchain.retrievers import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
import logging
import os
import backoff
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

os.environ['OPENAI_API_KEY'] = "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"

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
# load blog post
loader = WebBaseLoader("https://blog.langchain.dev/langchain-prompt-hub/")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

question = "What can we do with Langchain hub?"
llm = ChatOpenAI(temperature=0)


@retry(wait=wait_random_exponential(min=60, max=160), stop=stop_after_attempt(2))
def run_openai():
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), llm=llm)
    unique_docs = retriever_from_llm.get_relevant_documents(query=question)
    len(unique_docs)


if __name__ == '__main__':
    run_openai()
