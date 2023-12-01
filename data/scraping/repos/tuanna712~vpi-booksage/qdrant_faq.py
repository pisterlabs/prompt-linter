import os
import pandas as pd
import streamlit as st
from underthesea import word_tokenize
from langchain.schema import Document
from langchain.vectorstores import Qdrant
from langchain.embeddings import CohereEmbeddings
from qdrant_client import QdrantClient

def qdrant_faq_uploader(file_path, lang):
    class FAQdrant():
        def __init__(self, qdrant_url, qdrant_api_key, file_path):
            #Qdrant Key
            self.qdrant_url = qdrant_url
            self.qdrant_api_key = qdrant_api_key
            self.file_path = file_path

            #Embedding using Cohere Multilingual
            self.embeddings = CohereEmbeddings(model="multilingual-22-12", 
                                            cohere_api_key=os.environ['COHERE_API_KEY'])

        def read_file(self):
            #Load data from file 
            df = pd.read_excel(self.file_path)
            #Remove all null answer
            self.df = df.dropna(subset=['Trả lời'])
            self.question_list = self.df['Câu hỏi'].tolist()
            self.ans_list = self.df['Trả lời'].tolist()

        def langchain_docs(self):
            #Create Langchain Document including Question-Answer
            self.question_list_2 = self.question_list
            self.fn_question_list = []
            self.vi_processed_question_list = []
            self.ids = []
            for i in range(len(self.question_list)):
                self.ids.append(i)
                single_question = Document(page_content=self.question_list[i],
                                        metadata={'answer': self.ans_list[i], 
                                                    'n_question': i})
                self.fn_question_list.append(single_question)
                # Preprocessing with Underthesea
                self.question_list_2[i] = word_tokenize(self.question_list[i], format="text")
                vi_single_question = Document(page_content=self.question_list_2[i],
                                        metadata={'answer': self.ans_list[i], 
                                                    'n_question': i})
                self.vi_processed_question_list.append(vi_single_question)

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
            self.read_file()
            self.langchain_docs()
            self.upload_qdrant(self.fn_question_list, 'faq')
            if lang == 'vi':
                self.upload_qdrant(self.vi_processed_question_list, 'faqVieProcessed')
            print('Uploaded Documents to Qdrant in 2 Collection faq and faqVieProcessed')
    
    FAQdrant(st.session_state.qdrant_url, 
             st.session_state.qdrant_api_key, 
             file_path=file_path,
             ).processing()
    
    
# Upload FAQ DB to CHAT FACTs
def upload_chat_faq(df, collection_name):
    df = df.dropna(subset=['Trả lời'])
    embeddings = CohereEmbeddings(model="multilingual-22-12", 
                                cohere_api_key=os.environ['COHERE_API_KEY'])
    qdrant_url = os.environ['QDRANT_URL']
    qdrant_api_key = os.environ['QDRANT_API_KEY']
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    with st.spinner('Processing...'):
        # Preparing Documents before upload to Qdrant
        question_list = df['Câu hỏi'].tolist()
        answer_list = df['Trả lời'].tolist()
        ids = df.reset_index(drop=True).index.tolist()
        chunks = []
        for i in range(len(question_list)):
            chunk = Document(page_content=question_list[i], 
                                metadata={'answer': answer_list[i], 
                                        'type': 'FAQ'})
            chunks.append(chunk)
        
        # Upload and overwrite docs to Qdrant using Langchain Qdrant
        client.delete_collection(collection_name=collection_name)
        vdatabase = Qdrant.from_documents(documents=chunks,
                                        ids=ids,
                                        embedding=embeddings, 
                                        url=qdrant_url, 
                                        # prefer_grpc=True, 
                                        api_key=qdrant_api_key, 
                                        collection_name=collection_name,
                                        )
        st.success("Facts vector database created/updated")