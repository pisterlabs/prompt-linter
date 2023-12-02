import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from typing import List, Dict, Union
from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain import FAISS
from langchain.schema import Document

class KnowledgeBase:
    def __init__(
            self,
            file_dir: str
    ) -> None:
        """
        Creates a knowledge base for Agent.

        Args:
            file_dir (str): The path to the directory containing the documents to load.

        """
        self.path = file_dir
        self.embeddings = HuggingFaceHubEmbeddings(
            huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        )
        chunks = self.load_documents(self.path)
        embeds = self.create_embeddings(chunks[0])
        self.vector_store = self.create_vectorstore(chunks[0], embeds)
        
    def load_documents(
        self,
        docs_directory_path: str
    ) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
        """
        Load documents from a directory and return a list of dictionaries containing the name of each document and its chunks.

        Args:
            docs_directory_path (str): The path to the directory containing the documents to load.

        Returns:
            List[Dict[str, Union[str, List[Dict[str, str]]]]]: A list of dictionaries containing the name of each document and its chunks.
        """

        result = []

        for file_name in os.listdir(docs_directory_path):
            file_path = os.path.join(docs_directory_path, file_name)

            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(file_path=file_path)
            else:
                loader = UnstructuredFileLoader(file_path=file_path)

            document = loader.load()

            text_splitter = CharacterTextSplitter(
                separator=" ",
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len,
            )

            chunks = [
                {"chunk_" + str(i + 1): chunk.page_content}
                for i, chunk in enumerate(text_splitter.split_documents(documents=document))
            ]

            # Add document name and chunked data to result list
            file_name = os.path.splitext(file_name)[0]
            result.append({"name": file_name, "chunks": chunks})

        return result
    
    def create_embeddings(self, documents) -> List[Embeddings]:
        all_embeddings: list[Embeddings] = []
        texts: list[str] = []
        for doc in documents['chunks']:
            for key, value in doc.items():
                texts.append(value)
                break

        embeddings_list = self.embeddings.embed_documents(texts)

        all_embeddings.extend(embeddings_list)

        return all_embeddings
    
    def create_vectorstore(self, documents, embeds) -> FAISS:
        loaded_embeddings = embeds

        texts: list = []

        for doc in documents['chunks']:
            for key, value in doc.items():
                texts.append(value)
                break

        # Combine the texts and embeddings into a list of tuples
        text_embedding = list(zip(texts, loaded_embeddings))

        # Create a FAISS object from the embeddings and text embeddings
        faiss = FAISS.from_embeddings(embedding=self.embeddings, text_embeddings=text_embedding)

        return faiss

    def create_similarity_search_docs(
        self,
        query: str,
    ) -> List[Document]:
        """
        This function takes in three arguments: query, huggingfacehub_api_token, and path_to_vectorstore.
        It returns a list of documents that are most similar to the query.

        Parameters:
            - query (str): The query string.
            - huggingfacehub_api_token (str | None): The Hugging Face Hub API token.
            - path_to_vectorstore (str): The path to the vectorstore file.

        Returns:
            - List[Document]: A list of documents that are most similar to the query.
        """

        answer_docs = self.vector_store.similarity_search(query, k=4)
        return answer_docs[0].page_content

# know = KnowledgeBase('/home/nnpy/Downloads/Retrival_files')
# print(know.create_similarity_search_docs("what is LSTM"))