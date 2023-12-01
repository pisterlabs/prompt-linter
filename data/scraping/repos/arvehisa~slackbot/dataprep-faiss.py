import pickle
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import FAISS

# split docs
loader = PyPDFDirectoryLoader("./data/")

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# text chunks are saved as docs.pkl
with open('docs.pkl', 'wb') as f:
    pickle.dump(docs, f)

# embedding data
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
db = FAISS.from_documents(docs,bedrock_embeddings)
db.save_local("faiss_index")

