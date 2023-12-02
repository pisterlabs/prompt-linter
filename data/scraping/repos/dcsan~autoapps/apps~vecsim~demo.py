# Document loader
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

# Store 
from langchain.vectorstores import Chroma
# Using embedding models from OpenAI
# from langchain.embeddings import OpenAIEmbeddings
# vectorstore = Chroma.from_documents(documents=all_splits,embedding=OpenAIEmbeddings())
# Using local embedding models
from langchain.embeddings import HuggingFaceEmbeddings
vectorstore = Chroma.from_documents(
    documents=all_splits, 
    embedding=HuggingFaceEmbeddings(),
    persist_directory="./chroma_store"
)


question = "What are the approaches to Task Decomposition?"
print(question)
docs = vectorstore.similarity_search(question)
len(docs)
# print(docs)
for doc in docs:
    print(doc)
    print("\n")
