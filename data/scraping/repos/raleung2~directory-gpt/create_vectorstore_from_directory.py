'''
creates the vectorestore database for directory's documents
'''
#pylint: disable = too-few-public-methods, line-too-long

import ssl
import nltk
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

class BaseScript():
    '''
    disable ssl check to download punkt
    '''
    def __init__(self):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('punkt')

class DataConnection(BaseScript):
    '''
    tokenizes and adds embedding function to imported directory, saves embedding to vectorstore
    '''
    def __init__(
            self,
            document_directory:str,
            chroma_db_directory:str,
            embedding_function_model_name:str="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ):
        print("[1/5] Initializing nltk...")
        super().__init__()
        self.document_directory = f"./{document_directory}"
        self.chroma_db_directory = f"./chromadb/{chroma_db_directory}"
        self.embedding_function_model_name = embedding_function_model_name
        self.num_docs = 0

    def load_documents(self):
        '''
        load documents from desired directory
        '''
        loader = DirectoryLoader(self.document_directory, show_progress=True)
        docs = loader.load()
        return docs

    def create_chunks(self):
        '''
        creates chunks from tiktioken tokenizer
        '''
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
          chunk_size=400,
          chunk_overlap=0
          )
        print("[2/5] Loading documents...")
        chunks = text_splitter.split_documents(self.load_documents())
        print("[3/5] Chunks created...")
        return chunks

    def create_embeddings(self):
        '''
        creates embeddings with embedding function and saves to chromadb vectorstore
        '''
        embedding_function = SentenceTransformerEmbeddings(
            model_name=self.embedding_function_model_name
            )
        database = Chroma.from_documents(
            documents = self.create_chunks(),
            embedding = embedding_function,
            persist_directory=self.chroma_db_directory
            )
        print("[4/5] Chroma db created...")
        database.persist()
        print(f"[5/5] Chroma db saved to '{self.chroma_db_directory}'...")
        print("done!")

instance = DataConnection(
    document_directory="test-docs", #type folder only
    chroma_db_directory="test-1", #will create folder to save vectordb within root folder chromadb
    embedding_function_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
instance.create_embeddings()
