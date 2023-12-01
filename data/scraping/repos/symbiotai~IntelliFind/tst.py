from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

persist_directory = "db"
def indexText(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    doc = Document(page_content=text)
    docs = text_splitter.split_documents([doc])
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(
        documents=docs,
        persist_directory=persist_directory,
        embedding=embeddings)
    db.persist()

def search(prompt):
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return db.similarity_search(prompt)
