from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

class KubernetesIndex():
    vectordb: Chroma
    persistence_path: str
    doc_url: str
    def __init__(self, doc_url: str, persistence_path: str = "k8s_index"):
        self.doc_url = doc_url
        self.persistence_path = persistence_path
        self.vectordb = Chroma(persist_directory=persistence_path, embedding_function=OpenAIEmbeddings())
        
    def update_index(self):
        loader = OnlinePDFLoader(self.doc_url)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
        )
        split_docs = text_splitter.split_documents(loader.load())
        embedding = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(documents=split_docs, embedding=embedding, persist_directory=self.persistence_path)
