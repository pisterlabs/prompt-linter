# https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/Ask%20A%20Book%20Questions.ipynb used as reference

from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import pinecone
import openai
import time

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(client=OPENAI_API_KEY)

if PINECONE_API_KEY is not None and PINECONE_API_ENV is not None:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# loads pdf and splits into chunks then uploads to pinecone
def pdf_ingestion(url):
    pdf_loader = OnlinePDFLoader(url)
    data = pdf_loader.load()
    print (f'{len(data)} documents')
    print (f'{len(data[0].page_content)} characters')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    print (f'{len(chunks)} chunks to be uploaded')

    if PINECONE_API_KEY is not None and PINECONE_API_ENV is not None:
        for i, chunk in enumerate(chunks):
            Pinecone.from_texts([chunk.page_content], embeddings, index_name=PINECONE_INDEX_NAME)
            # probably overkill for this case, but we want to be nice to the API
            time.sleep(0.25)

# palantir 10Q
pdf_ingestion("https://d18rn0p25nwr6d.cloudfront.net/CIK-0001321655/fdffdcae-8f15-4011-b30b-b7ede07cf82c.pdf")