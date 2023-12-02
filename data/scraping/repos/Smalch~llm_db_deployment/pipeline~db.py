from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
import torch
device = 0 if torch.cuda.is_available() else -1  # set to GPU if available
device = 'auto'



class Db:
    def __init__(self):
        self.db_path = './db'
        model_name = 'e5-large-v2'
        model_path = f'./models/{model_name}'
        model_kwargs = {
            'device': 'cuda:0'
        }
        self.k = 2
        self.embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs=model_kwargs)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        self.reload_docs_with() #here should be load_db, but because I don't load to github my db, I put reload

    # @property
    # def retriever(self):
    #     return self.retriever

    #
    # def embed_doc(self):
    def add_document(self, path):
        loader = PyPDFLoader(path)
        doc = loader.load()
        docs = self.text_splitter.split_documents(doc)
        self.vectorDB.add_documents(docs)
        self.vectorDB.persist()

    def embed_dir(self):
        loader = DirectoryLoader('./Training_materials', glob='./*.pdf', loader_cls=PyPDFLoader)
        return loader.load()


    def reload_docs_with(self):
        loader = DirectoryLoader('./Training_materials/', glob="./*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        texts = self.text_splitter.split_documents(documents)
        self.vectorDB = None
        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=self.embeddings,
                                         persist_directory=self.db_path)

        vectordb.persist()
        self.vectorDB = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        self.retriever = self.vectorDB.as_retriever(search_kwargs={'k': self.k})

    def load_db(self):
        self.vectorDB = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        self.retriever = self.vectorDB.as_retriever(search_kwargs={'k': self.k})