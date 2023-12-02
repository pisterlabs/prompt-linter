import os 
import uuid

import json 
from langchain.vectorstores.faiss import FAISS


from langchain.tools import Tool
from langchain.document_loaders.json_loader import JSONLoader
from langchain.chains import ConversationalRetrievalChain

from pydantic.v1 import BaseModel
from pydantic.v1 import Field
from typing import Union, Tuple, Dict, List
from typing import Optional, Type

from leadgen.llms.base import BaseLLM
from .adapters.MockAdapter import MockAdapter

class UsersDatabase:
    EMBED_SAVE_DIR = "embeddings_user"
    EMBED_SAVE_INDEX = "user_embeddings"
    USERS_SAVE_DIR = "user_documents"

    def __init__(self, provider: BaseLLM, persist_directory = os.path.join("data", "users"), ) -> None:

        #Store our provider
        self.provider = provider

        #Load our vectorstore
        self.persist_dir = str(persist_directory)

        self.embeddings_dir = os.path.join(persist_directory, self.EMBED_SAVE_DIR)
        self.users_dir       = os.path.join(persist_directory, self.USERS_SAVE_DIR)

        #Create the directory if it doesn't exist
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.users_dir, exist_ok=True)

        _, self.vectorstore = self.load_vectorstore()


    def load_vectorstore(self):
        if os.path.exists(os.path.join(self.embeddings_dir, f'{self.EMBED_SAVE_INDEX}.faiss')):
            return True, FAISS.load_local(self.embeddings_dir, index_name=self.EMBED_SAVE_INDEX, embeddings=self.provider.get_embeddings())

        return False, FAISS.from_texts(["Dummy text."], self.provider.get_embeddings()) 

    def save_vectorstore(self):
        self.vectorstore.save_local(self.embeddings_dir, index_name=self.EMBED_SAVE_INDEX)

    def get_retriever(self, k = 5):
        return self.vectorstore.as_retriever(k = k)

    def get_qa_chain(self):
        llm = self.provider.get_llm()
        qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever= self.userdb.get_retriever(), return_source_documents=True)
        return qa 
    
    def get_llm(self):
        return self.provider.get_llm()