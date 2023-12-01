import os
import getpass

from openai import Embedding

os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('after-load/pdf.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=50)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OpenAIEmbeddings())

# Save the embeddings to a JSON file.
Embedding.to_file("embeddings.json", format="json")


query = "What should I invest in"
docs = db.similarity_search(query)
print(docs[0].page_content)