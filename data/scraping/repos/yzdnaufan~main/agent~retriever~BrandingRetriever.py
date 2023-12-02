from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

# Loader
b1 = PyPDFLoader("./doc/branding-1.pdf")
b2 = PyPDFLoader("./doc/branding-2.pdf")

# Split and load documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
splits = text_splitter.split_documents(b1.load())
splits2 = text_splitter.split_documents(b2.load())


for i in splits2:
    splits.append(i)

# Retrievers
branding_retriever = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings()).as_retriever()

brand_retriever = { "branding": branding_retriever}


# # Test
# print(chain.run("What is marketing?"))
# print(chain.run("What is marketing strategy?"))
# print(chain.run("What is coca cola marketing strategy?"))
# print(chain.run("What is coca cola marketing strategy?"))
# print(chain.run("What is coca cola marketing strategy?"))
# print(chain.run("What is coca cola marketing strategy?"))
