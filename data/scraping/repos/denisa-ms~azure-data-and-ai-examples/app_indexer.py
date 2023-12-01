from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import openai
import os

#load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")

OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_ADA_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_ADA_EMBEDDING_MODEL_NAME")

#init Azure OpenAI
openai.api_type = "azure"
openai.api_version = OPENAI_DEPLOYMENT_VERSION
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY

if __name__ == "__main__":
    embeddings=OpenAIEmbeddings(deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                                model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
                                openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                                openai_api_type="azure",
                                chunk_size=1)
    dataPath = "./data/documentation/"
    fileName = dataPath + "azure-azure-functions.pdf"

    #use langchain PDF loader
    loader = PyPDFLoader(fileName)

    #split the document into chunks
    pages = loader.load_and_split()

    #Use Langchain to create the embeddings using text-embedding-ada-002
    db = FAISS.from_documents(documents=pages, embedding=embeddings)



    #save the embeddings into FAISS vector store
    db.save_local("./dbs/documentation/faiss_index")