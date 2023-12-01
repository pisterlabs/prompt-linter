from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from variables import file_name, file_path
import pickle

# loader = OnlinePDFLoader(file_path)
loader = UnstructuredPDFLoader(file_name) # Default method via a downloaded file in local directory
data = loader.load()

# print(f"you have {len(data)} document/s")
# print(f"you have {len(data[0].page_content)} Characters")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=10)
texts = text_splitter.split_documents(data)
print(f"Splitting resulted in {len(texts)} Documents")
embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_documents(texts, embeddings)

with open("vectorstore.pkl", "wb") as f:
        pickle.dump(docsearch, f)