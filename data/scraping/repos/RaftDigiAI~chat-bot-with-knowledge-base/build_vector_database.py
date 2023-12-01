from typing import List

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.weaviate import Weaviate
from pandas import read_excel

from src.config.config import OPENAI_API_KEY, WEVIATE_URL


def build_and_get_database():
    documents: List[Document] = []
    file_content = read_excel("./data.xlsx")

    for row in file_content.iterrows():
        documents.append(
            Document(
                page_content=row[1]["content"],
                metadata={
                    "source": row[1]["source"],
                    "name": row[1]["name"],
                    "price": row[1]["price"],
                },
            )
        )

    print(f"Creating index in Weaviate at {WEVIATE_URL}")
    db = Weaviate.from_documents(
        documents,
        OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        weaviate_url=WEVIATE_URL,
    )
    return db
