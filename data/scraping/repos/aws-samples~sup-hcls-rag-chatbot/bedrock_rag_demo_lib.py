__import__('pysqlite3')
import os
import json
import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import JSONLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import Chroma
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate

credentials_profile_name = "ENTER CREDENTIAL PROFILE NAME" #sets the profile name to use for AWS credentials (if not the default)
region_name = "ENTER REGION" #sets the region name (if not the default)
endpoint_url = "https://bedrock-runtime.us-west-2.amazonaws.com" #sets the Bedrock endpoint.  If not using us-west-2 region, adjust this to the correct region

embeddings = BedrockEmbeddings(
    credentials_profile_name=credentials_profile_name,
    region_name=region_name, 
    endpoint_url=endpoint_url,
    model_id="amazon.titan-embed-text-v1"
) #create a Titan Embeddings client

index = Chroma(persist_directory="./db", embedding_function=embeddings)
index = VectorStoreIndexWrapper(vectorstore=index)

def get_memory(): #create memory for this chat session
    
    memory = ConversationBufferWindowMemory(ai_prefix="Assistant", human_prefix="Human", memory_key="chat_history", return_messages=True) #Maintains a history of previous messages
    
    return memory

def get_llm():
    
    model_kwargs =  { 
        "max_tokens_to_sample": 512,
        "temperature": 0, 
        "top_k": 250, 
        "top_p": 1, 
        "stop_sequences": ["\n\nHuman:"] 
    }
    
    llm = Bedrock(
        endpoint_url=endpoint_url,
        model_id="anthropic.claude-instant-v1", #use the Anthropic Claude model
        model_kwargs=model_kwargs) #configure the properties for Claude
    
    return llm
  
def get_rag_chat_response(input_text, memory): #chat client function
    
    llm = get_llm()
    
    input_text = f"""
    
        You are an administrative assistant for a healthcare insurance company.  You have access to a set of a patient's doctor encounters and related notes. 
        
        With that in mind, answer the following QUESTION:
        
        QUESTION:
        {input_text}
    
        """
        
    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, index.vectorstore.as_retriever(), memory=memory, verbose=True) # Uses memory for chat history
    chat_response = conversation_with_retrieval({"question": input_text}) #pass the user message and summary to the model
    chat_response = chat_response['answer']
    return(chat_response)