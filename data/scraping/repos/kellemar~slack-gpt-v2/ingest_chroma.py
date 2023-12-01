import os
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader, GitbookLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
import tiktoken
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
import time
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.document_loaders import ConfluenceLoader

load_dotenv()
embeddings = OpenAIEmbeddings()

CSV_FILE = './games_table.csv'
CSV_FILE_ARTICLES = './articles.csv'

st = time.time()

# Declare text_splitter to be used and what chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=64)

'''
# Convert CSV to Pandas Dataframe and add to ChromaDB collection
csv_loader = CSVLoader(CSV_FILE, source_column="source")
df = pd.read_csv(CSV_FILE)
loader = DataFrameLoader(df, page_content_column="content")
documents = loader.load()
documents = text_splitter.split_documents(documents)
persist_directory = 'db'
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory, collection_name="articles")
vectorstore.persist()


# Load CSV into ChromaDB
article_csv_loader = CSVLoader(CSV_FILE_ARTICLES, source_column="slug")
article_documents = article_csv_loader.load()
article_documents = text_splitter.split_documents(article_documents)
#vectorstore.add_documents(article_documents, embeddings=embeddings, collection_name="articles", metadata_column="title")
vectorstore = Chroma.from_documents(article_documents, embeddings, persist_directory=persist_directory, collection_name="articles")
vectorstore.persist()


# Load Gitbook into ChromaDB
gitbook_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
loader = GitbookLoader(os.environ["GITBOOK_URL"], load_all_paths=True)
all_pages_data = loader.load()
avg_documents = gitbook_text_splitter.split_documents(all_pages_data)
#vectorstore.add_documents(avg_documents, embeddings=embeddings, collection_name="avocado")
vectorstore = Chroma.from_documents(avg_documents, embeddings, persist_directory=persist_directory, collection_name="avocado")
vectorstore.persist()
'''