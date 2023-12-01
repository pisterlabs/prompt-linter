import os;
openai_api_key = os.environ.get('OPENAI_API_KEY')

# load syktw
# from langchain.document_loaders import UnstructuredHTMLLoader
# loader = UnstructuredHTMLLoader("books/syktw.html")
# book = loader.load()

# load Finnegan's
from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader("books/finnegans-wake")
book = loader.load()

# split it into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(book)

# create the vector store
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())