# Import Libraries
import os
import openai
import langchain
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


# Constants
CHUNK_SIZE = 2000
OPENAI_API_KEY = "YOUR-KEY"
PINECONE_API_KEY = "YOUR-KEY"
PINECONE_ENV = "YOUR-ENVIRONMENT"
INDEX_NAME = "YOUR-INDEX-NAME"


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def split_text(text, chunk_size=CHUNK_SIZE):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def setup_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)


def create_embeddings(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=0, length_function=len
    )
    notes = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return Pinecone.from_texts(
        [t.page_content for t in notes], embeddings, index_name=INDEX_NAME
    )


def run_chain(query, notes_docsearch, chain_type="stuff"):
    docs = notes_docsearch.similarity_search(query)
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type=chain_type)
    return chain.run(input_documents=docs, question=query)


def main():
    setup_pinecone()
    text = read_file("/content/notes.tex")
    notes_docsearch = create_embeddings(text)

    queries = [
        # Add your queries here
    ]

    for query in queries:
        print(run_chain(query, notes_docsearch))

    topics = [
        # Add your topics here
    ]

    for topic in topics:
        query = "What is " + topic + " and the definition and theorem?"
        print(run_chain(query, notes_docsearch))


if __name__ == "__main__":
    main()
