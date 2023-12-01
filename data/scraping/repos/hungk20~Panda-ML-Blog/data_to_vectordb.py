from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


# (1) Document Loading
loader = TextLoader("./born_pink_world_tour.md")
docs = loader.load()

# (2) Document Splitting
chunk_size = 150
chunk_overlap = 0
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""],
)
split_docs = text_splitter.split_documents(docs)

# (3) Store document and embeddings to vector store
embedding = OpenAIEmbeddings()
vectordb = FAISS.from_documents(
    documents=split_docs,
    embedding=embedding,
)
DATA_STORE_DIR = "data_store"
vectordb.save_local(DATA_STORE_DIR)
