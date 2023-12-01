from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

def get_chroma_db():

    load_dotenv()
    embedding_model = OpenAIEmbeddings()
    doc1 = Document(page_content="1 + 1 does not equal 4", metadata={"source": "source_a"})
    doc2 = Document(page_content="France is an interesting country", metadata={"source": "source_b"})
    docs = [doc1, doc2]
    # if db exists on disk, load it from disk, else create it
    if os.path.exists("./chroma_db_test"):
        db = Chroma(persist_directory="./chroma_db_test", embedding_function=embedding_model)
    else:
        db = Chroma.from_documents(docs, embedding_model, persist_directory="./chroma_db_test")
        db.persist()

    return db



