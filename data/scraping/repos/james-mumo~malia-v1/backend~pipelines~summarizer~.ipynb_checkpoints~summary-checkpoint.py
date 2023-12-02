#Loader
from langchain.schema import Document

# Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Model
from langchain.chat_models import ChatOpenAI

# Embedding Support
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

from langchain.prompts import PromptTemplate
# Summarizer, use for map reduce
from langchain.chains.summarize import load_summarize_chain

# Data Science
import numpy as np
from sklearn.cluster import KMeans

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# llm = ChatOpenAI()

# tokens = llm.get_num_tokens(text)

# print(tokens)

with open("huberman_transcripts.txt", "r") as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '\t'],
    chunk_size=1500,
    chunk_overlap=300
)

docs = text_splitter.create_documents([text])

embeddings = OpenAIEmbeddings()
vectors = embeddings.embed_documents([d.page_content for d in docs])

num_clusters = 7

# Perform K-means clustering 
kmeans = KMeans(n_init='auto' , n_clusters=num_clusters, random_state=42).fit(vectors)
