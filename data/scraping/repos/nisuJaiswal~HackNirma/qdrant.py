from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, PDFMinerLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from qdrant_client import QdrantClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import Filter
from qdrant_client.http import models as rest
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import uuid
import logging
from sentence_transformers import SentenceTransformer
from config import settings
import json

logging.basicConfig(level=logging.INFO, format='=========== %(asctime)s :: %(levelname)s :: %(message)s')


embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')

qa_chain = load_qa_chain(llm=OpenAI(openai_api_key=settings.openai_api_key, streaming=False), chain_type="stuff", verbose=False)
MetadataFilter = Dict[str, Union[str, int, bool]]

class qdrantDatabase():
    def __init__(self, qdrant_host:str, qdrant_api_key: str, prefer_grpc: bool):
        if qdrant_host == 'localhost':
            self.qdrant_client = QdrantClient(url="http://localhost:6333")
        else:
            self.qdrant_client= QdrantClient(
                url=qdrant_host,
                prefer_grpc=prefer_grpc,
                api_key=qdrant_api_key,
            )
        # we are using openai for embeddings
        self.embedding_model =  embedding_model
        self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()
        self.collection_name = "answerai"
        self.qdrant_client.recreate_collection(
            collection_name="answerai",
            vectors_config=VectorParams(size=self.embedding_size, distance=Distance.COSINE),
        ) 
        print(self)
        logging.info(f"Collection {self.collection_name} is successfully created.")

    def insert_into_index(self, filepath: str, filename: str):
        print(self)
        loader = PDFMinerLoader(filepath)
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=30)
        documents = text_splitter.split_documents(docs)

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        ids = [uuid.uuid4().hex for _ in texts]
        # encoding the text using openai
        vectors = self.embedding_model.encode(texts).tolist()
        payloads = []
        for i, text in enumerate(texts):
            if text is None:
                raise ValueError(
                    "At least one of the texts is None. Please remove it before "
                    "calling .from_texts or .add_texts on Qdrant instance."
                )
            metadata = metadatas[i] if metadatas is not None else None
            payloads.append(
                {
                    'page_content': text,
                    'metadata': metadata,
                }
            )

        # vectors = [vec.tolist() for vec in vectors]
        print("this are vectors")
        # print(vectors)
        # uploading the data to qdrant
        self.qdrant_client.upsert(
            collection_name="answerai",
            points=rest.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads
            ),
        )
        logging.info("Index update successfully done!")





    def generate_response(self, question: str):
        print("this is question:", question)
        print(self.similarity_search_with_score(query=question))
        relevant_docs = self.similarity_search_with_score(query=question)
        print("this are relevant docs:", relevant_docs)
        # return({'generated_response':'hello', 'relevant_docs':"relevant_docs"})
        return (qa_chain.run(input_documents=relevant_docs, question=question), relevant_docs)
    
        # Adopted from lanchain github
    def similarity_search_with_score(
        self, query: str, k: int = 5, filter: Optional[MetadataFilter] = None
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.
        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_model.encode(query)
        results = self.qdrant_client.search(
            collection_name="answerai",
            query_vector=embedding,
            query_filter=Filter(**filter) if filter else None,
            with_payload=True,
            limit=k,
        )
        print("this are results")
        print(results)
        for r in results:
            print(r)
            print()
        return [
            Document(
                 page_content=result.payload['page_content'],
                 metadata=result.payload['metadata']
            )
            for result in results
        ]
