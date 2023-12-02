# Importing important package 

import os
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms.bedrock import Bedrock

# Create llm object for Amazon Bedrock

def create_llm(model_name):
    
    model_kwargs =  { 
        "maxTokenCount": 1024, 
        "stopSequences": [], 
        "temperature": 0, 
        "topP": 0.9 
    }
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("DEMO_PROFILE_NAME"), 
        region_name=os.environ.get("DEMO_REGION_NAME"), 
        endpoint_url=os.environ.get("DEMO_ENDPOINT_URL"), 
        model_id=model_name,
        model_kwargs=model_kwargs) 
    
    return llm

def create_get_index(): 
    
    embeddings = BedrockEmbeddings(
        credentials_profile_name=os.environ.get("DEMO_PROFILE_NAME"), 
        region_name=os.environ.get("DEMO_REGION_NAME"), 
        endpoint_url=os.environ.get("DEMO_ENDPOINT_URL"), 
    ) 
    
    pdf_path = "redshift_table_definition.pdf" 

    loader = PyPDFLoader(file_path=pdf_path) 
    
    text_splitter = RecursiveCharacterTextSplitter( 
        separators=["\n\n", "\n", ".", " "], 
        chunk_size=1000, 
        chunk_overlap=100 
    )
    
    index_creator = VectorstoreIndexCreator( 
        vectorstore_cls=FAISS, 
        embedding=embeddings, 
        text_splitter=text_splitter, 
    )
    
    index_from_loader = index_creator.from_loaders([loader]) 
    
    return index_from_loader 
    
def call_rag_function(index, input_text): 

    model_name="amazon.titan-tg1-large"
    
    llm = create_llm(model_name)
    
    response_text = index.query(question=input_text, llm=llm) 
    
    return response_text



