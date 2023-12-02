from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

# Path: LLAMA2\__init__.py
# Read file and split into chunks
loader = PyPDFLoader("data/2Q23_Earnings_Release.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=50)
texts = text_splitter.split_documents(pages)

# # Load Sentence Transformers model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cpu'})

# # Encode documents
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local('vectorstore/db_faiss')