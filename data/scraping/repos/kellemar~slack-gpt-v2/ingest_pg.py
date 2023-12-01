import os
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader, GitbookLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.vectorstores.pgvector import PGVector
import tiktoken
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
import time
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.document_loaders import ConfluenceLoader
import psycopg2
load_dotenv()

#CONNECTION_STRING=os.environ["PG_CONNECTION"]

embeddings = OpenAIEmbeddings()

st = time.time()
CSV_FILE = './games_table.csv'
CSV_FILE_ARTICLES = './articles.csv'
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)

csv_loader = CSVLoader(CSV_FILE, source_column="source")
df = pd.read_csv(CSV_FILE)
df.fillna("", inplace=True)
loader = DataFrameLoader(df, page_content_column="content")
documents = loader.load()
documents = text_splitter.split_documents(documents)


# Load content from dataframe from CSV
'''
db = PGVector.from_documents(
    embedding=embeddings,
    documents=documents,
    collection_name="vectorstore_games",
    pre_delete_collection=True,
    distance_strategy="cosine",
)
df = pd.read_csv(CSV_FILE_ARTICLES)
df.fillna("", inplace=True)
loader = DataFrameLoader(df, page_content_column="content")
documents = loader.load()
documents = text_splitter.split_documents(documents)
articles_db = PGVector.from_documents(
    embedding=embeddings,
    documents=documents,
    collection_name="vectorstore_articles",
    pre_delete_collection=True,
    distance_strategy="cosine",
)'''

# Load Gitbook Content
'''
gitbook_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
loader = GitbookLoader(os.environ["GITBOOK_URL"], load_all_paths=True)
all_pages_data = loader.load()
avg_documents = gitbook_text_splitter.split_documents(all_pages_data)
gitbook_db = PGVector.from_documents(
    embedding=embeddings,
    documents=avg_documents,
    collection_name="vectorstore_gitbook",
    pre_delete_collection=True,
    distance_strategy="cosine",
)
'''

# Load Content from Confluence
'''
confluence_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)

loader = ConfluenceLoader(
    url="https://abc.atlassian.net/wiki/",
    api_key=os.environ["ATLASSIAN_KEY"],
    username=os.environ["ATLASSIAN_USER"],
)

documents = loader.load(page_ids=[123455,123456, 123457], include_attachments=False, limit=50)
confluence_split_doc = confluence_text_splitter.split_documents(documents)
confluence_db = PGVector.from_documents(
    embedding=embeddings,
    documents=confluence_split_doc,
    collection_name="vectorstore_confluence_tech",
    pre_delete_collection=True,
    distance_strategy="cosine",
)
'''
print(f"End of Document Index: {time.time() - st} seconds")
