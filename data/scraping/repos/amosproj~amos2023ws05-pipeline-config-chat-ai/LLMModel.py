from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate


# YourAPIKey = os.environ['OPENAI_API_KEY'] # This is the API key for OpenAI

load_dotenv() # Load the .env file

openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKey') # Get the API key from the .env file

llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=openai_api_key) # Load the LLM model


embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=openai_api_key) # Load the embeddings

# This is the root directory for the documents i want to create the RAG from
root_dir = '/Users/zainhazzouri/projects/RAG-Playground/core/src/sdk/python/rtdip_sdk/pipelines'
docs = [] # Create an empty list to store the docs

# Go through each folder to extract all the files
for dirpath, dirnames, filenames in os.walk(root_dir):

    # Go through each file
    for file in filenames:
        try:
            # Load up the file as a doc and split
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

docsearch = FAISS.from_documents(docs, embeddings) # Create the FAISS index
# source https://python.langchain.com/docs/integrations/vectorstores/faiss_async

# Get our retriever ready for the RAG or creating the chain
RAG = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
# source link for reference https://python.langchain.com/docs/use_cases/question_answering/document-context-aware-QA