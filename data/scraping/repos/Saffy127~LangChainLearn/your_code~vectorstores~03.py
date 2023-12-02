from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

loader = TextLoader("data/ai.txt")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=300,
  chunk_overlap=0
)

texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

persist_directory = 'db'

db = Chroma.from_documents(texts, embeddings, persist_directory)

db.persist()

query = "When was AI founded as an academic discpline?"

docs = db.similarity_search(query, k=1)

print(docs[0].page_content)
