# Import necessary modules and libraries
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
#from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from prompts import prompt
import argparse
import pandas as pd
import logging
import sys
import openai
import requests
import os

# Import sqlite3 as pysqlite3 and replace the sqlite3 module with pysqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Define the main function that processes the user's query
def main(query, path, isRag=False):
    
    # Load and split documents from the specified directory
    texts = loadAndSplitDocs(path)
    
    # Check and query the vector store for embeddings
    db, retriever = checkAndQueryVectorStore(texts, query) 
    
    # Run the user's query using the retriever
    result = runUserQuery(retriever, query, isRag)  
    
    # Print the result to the console
    print(result)

    # Write the result to a file named "initialConf.txt"
    filename = "initialConf.txt"
    if os.path.exists(filename):
        # If the file exists, append content to it
        with open(filename, 'w') as file:
            file.write(result)
    else:
        # If the file doesn't exist, create a new one
        with open(filename, 'x') as file:
            file.write(result)
                
    return result

# Function to load and split documents from a directory
def loadAndSplitDocs(path):
    # Create a directory loader and load documents
    loader = DirectoryLoader(path, glob="**/*.tf", use_multithreading=True)
    docs = loader.load()

    # Use a text splitter to split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        separators= ['"module ", "resource ", "variable "'],
        chunk_size = 150,
        chunk_overlap  = 20,
        length_function = len,
        add_start_index = True,
    )
    texts = text_splitter.split_documents(docs)
    return texts

# Function to check and query the vector store for embeddings
def checkAndQueryVectorStore(texts, query):
    # Create a Chroma vector store with OpenAI embeddings
    db = Chroma(persist_directory="./vectorStore", embedding_function=OpenAIEmbeddings())
    embeddingsList = db.get(include=['embeddings'])['embeddings']
    
    if  embeddingsList is None:
        logging.log(level=1, msg="INFO: Creating New VectorStore in the current directory")        
        db = Chroma.from_documents(texts, OpenAIEmbeddings(), persist_directory="vectorStore")        

    # Create a retriever for querying the vector store
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 2
    retriever.search_kwargs["maximal_marginal_relevance"] = False
    retriever.search_kwargs["k"] = 2
    return db, retriever

# Function to run the user's query using a chat model
def runUserQuery(retriever, query, isRag):
    # Create a ChatOpenAI model
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613",
              temperature=0.5,
              streaming=True, 
              max_tokens=1000)

    # Define a prompt template based on whether it's RAG or chat with your data use case
    if not isRag:
        extraction_prompt = PromptTemplate(input_variables=['context', 'question'],
                                           template="You are an expert at generating Terraform configurations for multiple cloud providers such as AWS, GCP, and Azure. Use the following context output either the terraform configuration or list of resources according to the Question. Don't make up any answer, if you don't know the answer just say I don't know. \n\n{context}\n\nQuestion: {question}\n Helpful Answer:")
    else:
        extraction_prompt = PromptTemplate(input_variables=['context', 'question' ],
                                           template=prompt)
    
    kwargs = {"prompt": extraction_prompt}
    
    # Create a RetrievalQA model for answering the user's query
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", chain_type_kwargs=kwargs, retriever=retriever) 

    # Run the user's query and return the result
    return qa.run(query)

# Set the OpenAI API key
openai.api_key = "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"

# Entry point of the script
if __name__ == "__main__":
    # Create an ArgumentParser object to parse command-line arguments
    parser = argparse.ArgumentParser(description="Script to process a query")

    # Define command-line arguments for query, path, and isRag
    parser.add_argument("query", type=str, help="The query to be processed")
    parser.add_argument("path", type=str, help="Path to the directory containing all documents.")
    parser.add_argument("isRag", type=str, help="True if it's RAG else it's chat with your data use case.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.query, args.path, args.isRag)
