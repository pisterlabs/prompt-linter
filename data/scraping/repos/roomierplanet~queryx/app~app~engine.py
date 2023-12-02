from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv

def update():
    load_dotenv()
    pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'), 
    environment=os.getenv('PINECONE_ENV') 
    )
    # index = pinecone.Index("queryx")
    # index.delete(deleteAll=True)
    loader = DirectoryLoader('../../YourDocs/')
    docs = loader.load()
    print('loaded')
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    Pinecone.from_documents([text for text in texts], embeddings, index_name = "queryx")

update()