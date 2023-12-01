 
import langchain
import pinecone
import os

from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


# Step 1: Load the PDF of the book from the web
pdf_url = "https://ebooksoff.xyz/AllNovelWorld.com/The-Price-Of-Tomorrow.pdf"
pdf_loader = OnlinePDFLoader(pdf_url)
pdf_text = pdf_loader.load()

# Step 2: Create a vector of the book using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(pdf_text)

# Step 3: Create a metadata dictionary with the book info
book_title = "The Price of Tomorrow"
book_text = pdf_text
book_author = "Jeff Booth"
book_publisher = "St. Martin's Press"
book_language = "English"
book_publication_year = 2020


metadata = {
    "title": book_title,
    "author": book_author,
    "language": book_language,
    "publication_date": book_publication_year
}

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

# Step 4: Connect to Pinecone
pinecone.init(
    api_key=st.secrets["PINECONE_API_KEY"],
    environment=st.secrets["PINECONE_API_ENV"]
)
index_name = "aitoshi"

# Step 5: Upload the book and metadata to a Pinecone index
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name, metadatas=[{"source": f"{book_title}-{i}-pl"} for i in range(len(texts))])
