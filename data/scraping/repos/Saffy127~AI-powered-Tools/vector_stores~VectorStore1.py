# Vector stores as a way to store text embeddings in a vector database
# We will us Chroma as our Vector db but we could also use PineCone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

#Define our text loader.
loader = TextLoader('data/ai.txt')

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=0)

texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(texts, embeddings)

# Instead of building a retriever we use the similarity_search function on our vector db directly to get the documents we want.

query = "When was AI founded as a discipline?"

docs = db.similarity_search(query, k=1)

print(docs[0].page_content)

