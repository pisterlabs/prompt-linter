# Import necessary libraries
import sqlite3
import requests
from concurrent.futures import ThreadPoolExecutor
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from dotenv import load_dotenv
import os
import PyPDF2
from io import BytesIO

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')

# Connect to SQLite database
conn = sqlite3.connect('links.db')
cursor = conn.cursor()

def process_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to retrieve {pdf_url}: {e}")
        return None
    
    pdf_stream = BytesIO(response.content)
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    pages_text = [page.extract_text() for page in pdf_reader.pages] # get the text from each page
    
    return pages_text

# Pinecone setup
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index_name = "langchain-retrieval-large"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric='cosine', dimension=1536)
index = pinecone.Index(index_name)

def create_and_store_embeddings(pages_text):
    embeddings_model = OpenAIEmbeddings()
    embeddings = embeddings_model.embed_documents(pages_text)
    item_mapping = {f"item-{i}": embedding for i, embedding in enumerate(embeddings)}
    index.upsert(vectors=item_mapping)

def process_and_store(pdf_url):
    pdf_url = pdf_url[0]
    print(f"Processing {pdf_url}")
    pages_text = process_pdf(pdf_url)
    if pages_text:
        print(f"Processed {len(pages_text)} pages from {pdf_url}")
        create_and_store_embeddings(pages_text)
    else:
        print(f"Failed to process {pdf_url}")

cursor.execute("SELECT pdf_links.url FROM pdf_links LEFT JOIN faulty_links ON pdf_links.url = faulty_links.url WHERE faulty_links.url IS NULL")
pdf_urls = cursor.fetchall()

with ThreadPoolExecutor() as executor:
    executor.map(process_and_store, pdf_urls)

conn.close()