from config_reader import ConfigReader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from components.PDFVectorStoreEnum import PDFVectorStoreEnum
import logging

logger = logging.getLogger(__name__)

class PDFVectorStore:
    embeddings = None
    db = None

    def __init__(self, openai_api_key, store_type=None):
        # Create OpenAI Embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        if (store_type is None):
            # get it from config
            config = ConfigReader()
            storage_type_str = config.get_value('vector_storage','store_type')
            store_type = PDFVectorStoreEnum(storage_type_str)

        match store_type:
            case PDFVectorStoreEnum.FAISS:
                texts = []
                texts.append("")
                text_embeddings = self.embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                self.db = FAISS.from_embeddings( 
                    text_embedding_pairs,               
                    self.embeddings
                )   
                logging.info("created the " + str(PDFVectorStoreEnum.FAISS) + " Vectorstore ")
            case PDFVectorStoreEnum.Chroma:
                raw_text = ''
                
                text_splitter = RecursiveCharacterTextSplitter(        
                    chunk_size = 100,
                    chunk_overlap  = 20,
                    length_function = len,
                )
                texts = text_splitter.split_text(raw_text)
                docs = text_splitter.create_documents(texts)
                self.db = Chroma(
                    "chroma_db_storage",
                    self.embeddings
                )
                logging.info("created the " + str(PDFVectorStoreEnum.Chroma) + " Vectorstore ")
            case _:
                self.db= DocArrayInMemorySearch.from_params(
                    self.embeddings
                )
                logging.info("created the " + str(PDFVectorStoreEnum.InMemory) + " Vectorstore ")
    
    def populate_db(self, docs):
        # Populate the database
        self.db.add_documents(docs)

    def get_retriever(self):
        # Creates the retriever
        return self.db.as_retriever()
