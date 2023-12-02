import os
import pandas as pd
import streamlit as st
from underthesea import word_tokenize
from langchain.schema import Document
from langchain.vectorstores import Qdrant
from langchain.embeddings import CohereEmbeddings

def qdrant_context_uploader(context, lang):
    class DocsQdrant():
        def __init__(self, qdrant_url, qdrant_api_key, file_path: str=None, context:str=None, lang:str=None):
            #Qdrant Key
            self.qdrant_url = qdrant_url
            self.qdrant_api_key = qdrant_api_key
            self.file_path = file_path  
            self.context = context      
            self.lang = lang
            #Embedding using Cohere Multilingual
            self.embeddings = CohereEmbeddings(model="multilingual-22-12", 
                                            cohere_api_key=os.environ['COHERE_API_KEY'])

        def read_file(self):
            # Load context from txt file
            with open(self.file_path, 'r') as f:
                _context = f.read()
            # Split to small chunks, delimited by '\n\n'
            self.context_list = _context.split('\n\n')

        def langchain_docs(self):
            #Create Langchain Document including Question-Answer
            self.context_docs = []
            self.vi_processed_context_list = []
            self.ids = []
            for i in range(len(self.context_list)):
                self.ids.append(i)
                single_context = Document(page_content=self.context_list[i], 
                                        metadata={'n_context': i})
                self.context_docs.append(single_context)
                
                # Preprocessing with Underthesea
                self.vi_context_list = self.context_list
                self.vi_context_list[i] = word_tokenize(self.vi_context_list[i], format="text")
                vi_single_context = Document(page_content=self.vi_context_list[i], 
                                        metadata={'n_context': i})
                self.vi_processed_context_list.append(vi_single_context)
            
        def upload_qdrant(self, docs, collection_name):
            #Upload Documents to Qdrant Online Storage
            Qdrant.from_documents(docs,
                                self.embeddings, 
                                ids = self.ids,
                                url=self.qdrant_url, 
                                api_key=self.qdrant_api_key, 
                                content_payload_key="page_content",
                                metadata_payload_key="metadata",
                                collection_name=collection_name,
                                )

        def processing(self):
            if self.context is not None:
                # check if context is a string or a list of strings
                if isinstance(self.context, str):
                    self.context_list = self.context.split('\n\n')
                elif isinstance(self.context, list):
                    self.context_list = self.context
            else:
                self.read_file()
            self.langchain_docs()
            self.upload_qdrant(self.context_docs, 'context')
            if self.lang == 'vi':
                self.upload_qdrant(self.vi_processed_context_list, 'contextVieProcessed')
            print('Uploaded Documents to Qdrant in 2 Collection context and contextVieProcessed')

    DocsQdrant(st.session_state.qdrant_url, 
             st.session_state.qdrant_api_key, 
             context=context,
             lang=lang).processing()
