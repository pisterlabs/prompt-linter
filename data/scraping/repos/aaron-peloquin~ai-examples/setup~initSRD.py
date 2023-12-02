from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter


loader = UnstructuredMarkdownLoader("data\srd\srd5-1-creative-commons.md")
text_splitter = MarkdownTextSplitter(chunk_size=512, chunk_overlap=64)
docs = text_splitter.split_documents(loader.load())
embeddings = HuggingFaceEmbeddings()
persist_directory = '.chromaDB/srd'

chromaDb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
chromaDb.persist()
