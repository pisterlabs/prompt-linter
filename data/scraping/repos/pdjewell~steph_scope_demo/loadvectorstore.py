import os
import streamlit as st
import openai
from langchain.vectorstores import Chroma 
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv


class VectorStoreLoader:

    def __init__(self,
                 vector_db_directory=os.path.join(os.getcwd(), 'vector_db'),
                 ):
        self.vector_db_directory = vector_db_directory


    # Load chroma db from file
    def load_chroma_db(self, persist_directory):

        # get openai api key from env
        load_dotenv()
        openai.api_key = os.environ.get('OPENAI_API_KEY')

        # Select embedding model
        model_name = 'text-embedding-ada-002'
        embeddings = OpenAIEmbeddings(model=model_name, 
                                    openai_api_key=openai.api_key)

        # Load vectorstore
        vectorstore = Chroma(persist_directory=persist_directory, 
                            embedding_function=embeddings)

        return vectorstore
    

    def load_all_chroma_db(self):

        vectorstores = {}
        for vector_db in os.listdir(self.vector_db_directory):
            print(vector_db)
            vector_db_path = os.path.join(self.vector_db_directory, vector_db)
            print(vector_db_path)
            vectorstore = self.load_chroma_db(vector_db_path)
            vectorstores[vector_db] = vectorstore

        print(f"Successfully loaded: {len(vectorstores)} vectorstores")

        return vectorstores
