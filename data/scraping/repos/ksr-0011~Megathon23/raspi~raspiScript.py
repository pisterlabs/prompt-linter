


from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import openai

import numpy as np
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone

from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
import keyboard


os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"




query =''
while True:
    try:
        query = input("Enter your query (Press 'Esc' to exit): ")
        

        text = "Hello this is test insert"
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function = len)
        docs = text_splitter.split_text(text)
        

   
        pinecone.init(      
	api_key='f2fd1af8-e3d0-494e-9188-ccaedca13bdd',      
	environment='gcp-starter'      
)      
        index_name = "medical1"
        index_exists = False

        indexes = pinecone.list_indexes()

        
        if index_name in indexes:
          index_exists = True
        
        
# response_data = {'message': 'query unsuccessfully', 'answer': "Please upload a File!"}
    

        embeddings = OpenAIEmbeddings(model_name="ada")
        embedding = embeddings.embed_query(docs[0])
        index = pinecone.Index(index_name)
        index = Pinecone.from_texts(docs, embeddings, index_name=index_name)
        

        k=3 
        score = False
        if score:
         similar_docs = index.similarity_search_with_score(query,k=k)
        else:
         similar_docs = index.similarity_search(query,k=k)
         similar_docs
        
        # model_name = "text-davinci-003"
        model_name = "gpt-3.5-turbo"
        # model_name = "gpt-4"
        llm = OpenAI(model_name=model_name)


        chain = load_qa_chain(llm, chain_type="stuff")

        answer =  chain.run(input_documents=similar_docs, question=query)



        print("Answer:", answer)
    except Exception as e:
        print("An error occurred:", str(e))


       
