import os
import csv
import chromadb
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

load_dotenv('../.env')
PERSIST_DIRECTORY = "../vector_db"
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, PERSIST_DIRECTORY)
CONDITIONS = "conditions"


settings = chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=DB_DIR,
    anonymized_telemetry=False,
)


def get_client():
    return chromadb.Client(settings=settings)


def create_health_conditions_qa_db():
    EMBEDDINGS = OpenAIEmbeddings()
    docs = []
    with open("../clean_data/ProcessedData.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            question, answer, focus = row
            if question == "Questions":
                # skip header
                continue
            text = f"{question}\n{answer}"
            doc = Document(
                page_content=text, metadata={"focus": focus, "question": question}
            )
            docs.append(doc)

    db = Chroma(
        collection_name=CONDITIONS,
        embedding_function=EMBEDDINGS,
        client_settings=settings,
        persist_directory=DB_DIR,
    )
    db.add_documents(documents=docs, embedding=EMBEDDINGS)
    db.persist()

    return db


def get_health_conditions_qa_db(client):
    EMBEDDINGS = OpenAIEmbeddings()
    collections = [col.name for col in client.list_collections()]
    if CONDITIONS in collections:
        return Chroma(
            collection_name=CONDITIONS,
            embedding_function=EMBEDDINGS,
            client_settings=settings,
            persist_directory=DB_DIR,
        )
    return create_health_conditions_qa_db()


def match_condition(client, inp):
    db = get_health_conditions_qa_db(client)
    docs = db.similarity_search(inp)
    return docs


def retreiver(client, inp):
    db = get_health_conditions_qa_db(client)
    retriever = db.as_retriever(search_type="mmr")
    docs = retriever.get_relevant_documents(inp)
    return docs


if __name__ == "__main__":
    # client = get_client()
    # time.sleep(5)
    text = """Back of my neck is hurting.
    I didn't have any injuries. I work as a software engineer.
    I have been experiencing this for 2-3 months."""
    # docs = match_condition(client, text)
    # for doc in docs:
    #     print(doc.metadata["focus"])
    #     print(doc.page_content)
    #     print("=========================")

    print("@" * 20)
    client = get_client()
    docs = retreiver(client, text)
    for doc in docs:
        print(doc.metadata["focus"])
        print(doc.page_content)
        print("=========================")
