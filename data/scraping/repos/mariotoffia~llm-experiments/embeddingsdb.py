import os
import shutil
import json
from typing import List
import uuid
from langchain.docstore.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.schema.embeddings import Embeddings

class EmbeddingsDb:
    """
    Embeddings database
    """
    chroma: Chroma
    embeddings_path: str = "./data/embeddings"
    embeddings: Embeddings
    search_type: str
    k: int

    def __init__(self,
                 embeddings: Embeddings,
                 search_type="similarity",
                 k=4,
                 ):
        """
        Constructor
        :param embeddings: The embeddings creator to use.
        """
        if not os.path.exists(self.embeddings_path):
            os.makedirs(self.embeddings_path)

        self.chroma = Chroma(
            embedding_function=embeddings,
            persist_directory=self.embeddings_path,
        )

        self.embeddings = embeddings
        self.search_type = search_type
        self.k = k

    def get_embeddings(self) -> Embeddings:
        return self.embeddings

    def as_retriever(self):
        """
        Return the Chroma object as a retriever
        :return: Chroma object
        """
        return self.chroma.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.k},
        )

    def embed(self, text: str) -> List[float]:
        """
        Embed a text
        :param text: Text to embed
        :return: List of floats
        """
        return self.embeddings.embed_query(text)

    def reset(self):
        """
        Reset the vector store by delete all files and recreating the directory
        where the embeddings are stored.
        :return:
        """
        if not os.path.exists(self.embeddings_path):
            return

        for item in os.listdir(self.embeddings_path):
            item_path = os.path.join(self.embeddings_path, item)

            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    def query_text(self, text: str) -> List[Document]:
        """
        Query the vector store for the given text
        :param text: Text to query
        :return: List of Document objects
        """
        docs = self.chroma.as_retriever().get_relevant_documents(query=text)

        seen_ids = set()
        result: List[Document] = []

        for doc in docs:
            if str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) not in seen_ids:
                result.append(doc)

                seen_ids.add(
                    str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)))

        return result

    def store_structured_data(self, docs: List[Document], id: str = None) -> bool:
        """
        Store structured data in the vector store
        :param docs:    List of Document objects
        :param id:      Optional id, of which it checks if already indexed and skips
                        if such is the case. 
        :return:        True if the data was stored, False if the data was skipped
        """
        if not os.path.exists(self.embeddings_path):
            os.makedirs(self.embeddings_path)

        id_path = os.path.join(self.embeddings_path, "indexed", id)

        if id is not None and os.path.exists(id_path):
            return False

        self.chroma.from_documents(
            documents=docs,            
            persist_directory=self.embeddings_path,
            embedding=self.embeddings,
        )

        # Mark id as already done
        if id is not None:
            os.makedirs(os.path.dirname(id_path), exist_ok=True)
            with open(id_path, "w") as f:
                f.write(id)

        return True
