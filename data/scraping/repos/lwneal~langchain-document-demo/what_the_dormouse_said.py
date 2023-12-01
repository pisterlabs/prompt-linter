import time
import os
import getpass
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

def green(text):
    return '\033[92m' + text + '\033[0m'


# Create the db
loader = TextLoader("alice.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
print('Created db with {} documents'.format(len(docs)))


# Ask questions
query = "What did the dormouse say?"
docs = db.similarity_search(query)
docs_and_scores = db.similarity_search_with_score(query)
print()
print(green("Query:"))
print(green(query))
time.sleep(2)
print(green("Best Answer:"))
print(docs_and_scores[0][0].page_content)
time.sleep(3)
print(green("Second-Best Answer:"))
print(docs_and_scores[1][0].page_content)

