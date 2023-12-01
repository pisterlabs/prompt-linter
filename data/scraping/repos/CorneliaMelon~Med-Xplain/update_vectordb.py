from utils import *
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
from langchain.vectorstores import Pinecone

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load() # this will load a list of documents
    # get number of diffrent pages in the PDF
    print(len(pages))
    trimmed_pages = pages[1:2] # Just the relevant pages
    page = trimmed_pages[0]
    return str(page.page_content[0:20000])

def split_text(text, max_characters=23250):
    text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "], 
            chunk_size=10000, 
            chunk_overlap=2200)
    docs = text_splitter.create_documents([text[:max_characters]])
    return docs

def update_vectordb(docs):
    pinecone.init(
    api_key="",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
    )
    index_name = "chatbot"
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)



