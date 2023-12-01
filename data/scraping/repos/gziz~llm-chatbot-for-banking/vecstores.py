from pathlib import Path
from typing import List

from langchain.docstore.document import Document

current_file_path = Path(__file__).resolve()
VECSTORE_DIR = current_file_path.parent / "chroma"


class Vecstores:
    def __init__(self) -> None:
        self.load_vecstores()

    def load_vecstores(self) -> None:
        from langchain.vectorstores import Chroma

        embeddings = self.get_embeddings_engine()

        self.profiles = {}
        for profile_path in VECSTORE_DIR.iterdir():
            profile = str(profile_path.stem)

            self.profiles[profile] = \
                Chroma(
                    persist_directory=str(profile_path),
                    embedding_function=embeddings,
                )


    def get_embeddings_engine(self):
        from langchain.embeddings import OpenAIEmbeddings

        embedding_engine = OpenAIEmbeddings(model="text-embedding-ada-002")
        return embedding_engine

    def similarity_search(self, q: str, profile: int) -> List[Document]:

        docs = self.profiles[profile].similarity_search(q, k=3)
        return docs
