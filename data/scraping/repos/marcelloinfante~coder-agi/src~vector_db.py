from random import random

import chromadb
from chromadb.utils import embedding_functions

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


class VectorDB:
    def __init__(self, config, llm, memory):
        self.llm = llm
        self.config = config
        self.memory = memory

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-ada-002"
        )

        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name="vector_db", embedding_function=openai_ef
        )

    def query(self, current_objective):
        query_text = current_objective["description"]
        query_results = self.collection.query(query_texts=[query_text], n_results=20)
        return query_results

    def update_collection(self):
        for file_path in self.memory.commited_files:
            try:
                if not ".gitignore" in file_path:
                    self.collection.delete(where={"path": f"./{file_path}"})

                    loader = TextLoader(self.config.path + file_path, encoding="utf-8")
                    docs = loader.load()

                    text_splitter = CharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=0
                    )
                    texts = text_splitter.split_documents(docs)

                    for text in texts:
                        self.collection.add(
                            ids=[str(int(random() * 10**14))],
                            documents=[text.page_content],
                            metadatas=[{"path": f"./{file_path}"}],
                        )

            except Exception as e:
                print(e)
                continue

        self.memory.commited_files = []
