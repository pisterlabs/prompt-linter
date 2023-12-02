from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
#comment out because we dont have to do it again after the first run.

"""
# Load our document.
loader = TextLoader('data/ai.txt')

documents = loader.load()

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)

texts = text_splitter.split_documents(documents)
"""

embeddings = OpenAIEmbeddings()

persist_directory = 'db'

#Now we specify where we want our data persisted to.

#Lets call the db directory

persisted_db = Chroma(embedding_function=embeddings, persist_directory=persist_directory)


query = "When was AI founded as an academic discipline?"

docs = persisted_db.similarity_search(query)

print(docs)
