
import os
import openai
from langchain.llms import AzureOpenAI

from dotenv import load_dotenv
from llama_index import GPTChromaIndex, TrafilaturaWebReader
import chromadb

load_dotenv()
# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

def create_embedding_store(name):
    chroma_client = chromadb.Client()
    return chroma_client.create_collection(name)

def query_pages(collection, urls, questions):
    docs = TrafilaturaWebReader().load_data(urls)
    index = GPTChromaIndex.from_documents(docs, chroma_collection=collection)
    for question in questions:
        print(f"Question: {question} \n")
        print(f"Answer: {index.query(question)}")

if __name__ == "__main__":
    url_list = ["https://supertype.ai", "https://supertype.ai/about-us"]
    questions = [
        "Who are the members of Supertype.ai", 
        "What problems are they trying to solve?",
        "What are the important values at the company?"
    ]

    collection = create_embedding_store("supertype")

    query_pages(
        collection,
        url_list,
        questions
    )
