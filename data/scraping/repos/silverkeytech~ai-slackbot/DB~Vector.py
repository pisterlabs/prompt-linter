from DB.Database import Database
from abc import ABC, abstractmethod

import qdrant_client
from langchain.vectorstores import Qdrant      
from qdrant_client import QdrantClient
from dotenv import find_dotenv, load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings

class VectorDB(Database):
    @abstractmethod
    def store(self, data):
        pass

    @abstractmethod
    def embed(self, data):
        pass

class QdrantVector(VectorDB):

    def __init__(self):
        embeddings_model_name="all-MiniLM-L6-v2"

        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

        self.Connect()

        load_dotenv(find_dotenv())

        self.github_vector_store= Qdrant(
            client= self.qdrant_client,
            collection_name= os.environ["QDRANT_COLLECTION_NAME_GITHUB"],
            embeddings=self.embeddings,
            )

        self.slack_attendance_vector_store= Qdrant(
            client= self.qdrant_client,
            collection_name= os.environ["QDRANT_COLLECTION_NAME_SLACK_ATTENDANCE"],
            embeddings=self.embeddings,
            )

        self.slack_users_vector_store= Qdrant(
            client= self.qdrant_client,
            collection_name= os.environ["QDRANT_COLLECTION_NAME_SLACK_USERS"],
            embeddings=self.embeddings,
        )

    def Connect(self):
        self.qdrant_client = QdrantClient(host='localhost', port=6333, grpc_port=6333)

    def getContext(self, question, choice):
        
        # contextUsers = self.slack_users_vector_store.similarity_search(question, k=2)
        # contextGithub = self.github_vector_store.similarity_search(question, k=2)
        # contextAttendance = self.slack_attendance_vector_store.similarity_search(question, k=5)

        if choice == 0:
            context = self.slack_users_vector_store.similarity_search(question, k=2)
        elif choice == 1:
            context = self.github_vector_store.similarity_search(question, k=2)
        elif choice == 2:
            context = self.slack_attendance_vector_store.similarity_search(question, k=5)
        
        return context
    
    def store(self, data):
        pass

    def embed(self, data):
        pass

    def Disconnect(self):
        pass
