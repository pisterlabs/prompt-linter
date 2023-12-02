import os
import streamlit as st
import tiktoken
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from underthesea import word_tokenize
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()

class DocProcessing():
    def __init__(self, 
                 raw_txt,
                 chunk_size:int=500,
                 chunk_overlap:int=100,
                 collection_name:str=None,
                 book_lang:str='en',
                 separators:list=['\n\n\n\n','\n\n\n','\n\n', '\n', ' ', ''],
                 ):
        
        ##Self Definition--------------------------
        self.data = raw_txt
        self.qdrant_url = os.environ['QDRANT_URL']
        self.qdrant_api_key = os.environ['QDRANT_API_KEY']
        self.embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=os.environ['COHERE_API_KEY'])
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.book_lang = book_lang
        self.separators = separators

        ##Processing...----------------------------
    def file_processing(self):
        if len(self.data)==1:
            self.data = self.data[0]
            self.text_splitting()
        else:
            self.chunks = self.data
        self.vi_processing()
        self.qdrant_vectorization()
        st.success('VectorDatabase!')
        
    def text_splitting(self):
        #Split to Smaller Texts
        spliter = RecursiveCharacterTextSplitter(
                                chunk_size=self.chunk_size,
                                chunk_overlap=self.chunk_overlap,
                                length_function=self.tiktoken_len,
                                separators=self.separators,
                                )
        self.chunks = spliter.split_text(self.data)
    def vi_processing(self):
        ##Process-MS WORD---------------------------
        if self.book_lang=='vi':
            self.chunks = [word_tokenize(self.chunks[i], format="text") 
                            for i in range(len(self.chunks))]
            
    def tiktoken_len(self, text):
        tokenizer = tiktoken.get_encoding('cl100k_base')
        tokens = tokenizer.encode(text, disallowed_special=())
        return len(tokens)

    def qdrant_vectorization(self):
        qdrant_url = os.environ['QDRANT_URL']
        qdrant_api_key = os.environ['QDRANT_API_KEY']
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        client.delete_collection(collection_name=self.collection_name)
        with st.spinner(text='Embedding...'):
            ids = [i for i in range(len(self.chunks))]
            self.vdatabase = Qdrant.from_texts(self.chunks,
                                            self.embeddings, 
                                            ids = ids,
                                            url=self.qdrant_url, 
                                            prefer_grpc=True, 
                                            api_key=self.qdrant_api_key, 
                                            collection_name=self.collection_name,
                                            )
        st.info("Qdrant Vectorized Chunks use on-cloud storage\n")
    