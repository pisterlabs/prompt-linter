from langchain.memory import ConversationBufferMemory
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

import os
# import numpy as np


def get_pdf_text(pdf_docs):
    
    text = ""
    for pdf in pdf_docs:
        print(f'Reading {pdf}')
        pdf_reader = PdfReader(f'Knowledge\data\{pdf}')
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks
    
def make_vectorstore(text_chunks):
    
    # embeddings = OpenAIEmbeddings()
    # test = np.load('savefile.npy',allow_pickle=True)
    # if (test):
    #     return test
    # embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    embeddings = OpenAIEmbeddings(deployment=os.getenv("OPENAI_DEPLOYMENT_NAME"),chunk_size=1,)
    print('Embedding started')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding = embeddings)
    # vectorstore=FAISS.load_local('vectorstore',embeddings)
    print('Vector store received')
    vectorstore.save_local('Knowledge\Vectorstore')
    # vs = np.array(vectorstore)
    # np.save('savefile.npy',vs)
    return vectorstore


    

            
def Develop():
    if len(os.listdir('Knowledge\Vectorstore'))<1:
        load_dotenv()
                
        raw_text = get_pdf_text(os.listdir('Knowledge\data'))

        text_chunks = get_chunks(raw_text)
        print(text_chunks[:1])
        make_vectorstore(text_chunks)
        print('Vector stores are created')
        
    else:
        print('Vector stores For Knowledge are already present')

    
    
    
    
    
# if __name__=='__main__':
#     Develop()