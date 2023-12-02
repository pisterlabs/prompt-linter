import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
loader = PyPDFLoader('./Psychotherapy.pdf')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(
  documents,
  embedding=OpenAIEmbeddings(),
  persist_directory='./data'
)

vectordb.persist()
