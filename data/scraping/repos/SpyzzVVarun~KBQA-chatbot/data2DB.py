import pinecone
from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import Pinecone
from constants import index_name, PINECONE_API_KEY, ENV, EMB_MODEL
from data_loading_utils import *

directory = '/content/demo'
documents = load_docs(directory)
docs = split_docs(documents)
embeddings = HuggingFaceBgeEmbeddings(model_name = EMB_MODEL)


# initialize pinecone
pinecone.init(
    api_key= PINECONE_API_KEY,
	environment=ENV
)
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)