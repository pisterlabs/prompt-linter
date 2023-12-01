from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.document_loaders import WebBaseLoader
from dotenv import load_dotenv
load_dotenv()

# Intenta ver si puedes cambiar el documento(s) que se cargan. Por ejemplo, un pdf.
loader = WebBaseLoader(["https://www.meetup.com/ia-generativa-sevilla/"])
websites = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20, separators=['\n', '. ', '?', '! '])
documents = text_splitter.split_documents(websites)

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(documents, embeddings, collection_name='meetup', persist_directory='./vectordb')

if __name__ == '__main__':
    query = "Who is the organizer of Generative IA Sevilla?"
    docs = db.similarity_search(query)
    print(docs)