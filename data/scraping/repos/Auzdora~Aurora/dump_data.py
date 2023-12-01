from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import NotionDirectoryLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from configer import ConfigLoader

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100,
    length_function = len,
    add_start_index = True,
)

class DataDumper:
    """
    DataDumper is a class that dump data from database to local, and load data from local to database.
    """
    def __init__(self, config_loader: ConfigLoader) -> None:
        """
        Args:
            config_loader (ConfigLoader): config loader
        """
        self.OPEN_AI_KEY = config_loader.get_api_key()
        self.OPEN_AI_BASE = config_loader.get_api_base()
        self.config_loader = config_loader

        self.text_spliter = RecursiveCharacterTextSplitter(
            chunk_size = 5000,
            chunk_overlap = 1000,
            length_function = len,
            add_start_index = True,
        )

        self.embedding_model = OpenAIEmbeddings(
            openai_api_key=self.OPEN_AI_KEY,
            openai_api_base=self.OPEN_AI_BASE
        )

        if self.config_loader.config["system"]["vector_db"]["localized"] is False:
            print("dumping your local data ...")
            raw_documents = []

            # data connection
            if self.config_loader.config["system"]["database"].get("notion_db"):
                loader = NotionDirectoryLoader("Notion_DB")
                notion_raw_docs = loader.load()
                raw_documents.append(notion_raw_docs)
                
            if self.config_loader.config["system"]["database"].get("mardown_db"):
                pass
            
            split_doc = self.text_spliter.split_documents(raw_documents[0])
            vector_db = FAISS.from_documents(split_doc, self.embedding_model)

            # split documents and add more documents
            for raw_doc in raw_documents[1:]:
                split_doc = self.text_spliter.split_documents(raw_doc)
                vector_db.add_documents(split_doc, self.embedding_model)

            self.dump_vector_db(vector_db)
            self.config_loader.config["system"]["vector_db"]["localized"] = True
            self.config_loader.write()
        
        self.store_path = self.config_loader.config["system"]["vector_db"]["store_path"]

    def dump_vector_db(self, vector_db):
        """
        dump vector db to local
        """
        vector_db.save_local("./")
    
    def get_vector_db(self) -> FAISS:
        """
        load vector db from local
        """
        return FAISS.load_local(self.store_path, embeddings=self.embedding_model)
