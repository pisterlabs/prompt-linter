import os
from util.config import Config

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import BSHTMLLoader, UnstructuredHTMLLoader
from langchain.document_loaders import JSONLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredPDFLoader

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

config = Config()

class Ringley_Database:

    def __init__(self, db_dir=None):
        if db_dir is None:
            self.db_dir = config.get_db_dir()
        else:
            self.db_dir = db_dir
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        # db_list is a list of the names of the embedding databases
        # faiss_list is a list of the names of the faiss databases
        if os.listdir(self.db_dir) == []:
            self.db_list = []
        else:
            self.db_list = os.listdir(self.db_dir)

        self.data_dir = config.get_data_dir()
        self.update_data_dir = config.get_update_data_dir()

        self.vector_dir = config.get_vector_dir()
        if not os.path.exists(self.vector_dir):
            os.mkdir(self.vector_dir)

        if os.listdir(self.vector_dir) == []:
            self.faiss_exists = False
        else:
            self.faiss_exists = True
        self.supported_data_types = ["txt", "csv", "json", "html", "md", "pdf"]

    def show_db_list(self):
        print("The embedded databases are:")
        for db_name in self.db_list:
            print(db_name)
    
    def show_existing_embeddings(self):
        print("The existing embeddings are:")
        for db_name in os.listdir(self.db_dir):
            print(db_name)
    
    def show_supported_data_types(self):
        print("The supported data types are:")
        print(self.supported_data_types)
    
    def show_data_loader(self):
        print("The data loaders are:")
        print(self.switcher)

    def check_existence(self, data_file):
        db_name = self.create_embeddingdb_name(data_file)
        if db_name in self.db_list:
            return True
        else:
            return False

    def set_OPENAI_API_KEY(self, config=config):
        # Set the OpenAI API key as an environment variable
        OPENAI_API_KEY = config.get_openai_key()
        if OPENAI_API_KEY is None:
            print("The OPENAI_API_KEY does not exists. Please set the OpenAI API key in the config file.")
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    def create_embeddingdb_name(self, filename, note=None):
        """
            INPUT: filename: str, note: str
            OUTPUT: name: str
        """
        if note is None:
            name = filename.split(".")[0]
        else:
            name = filename.split(".")[0] + "_" + note
        # Note: replace the special characters in the filename
        name = name.replace(" ", "_")
        name = name.replace("(", "_")
        name = name.replace(")", "_")
        name = name.replace(",", "_")
        name = name.replace("'", "_")
        name += ".pkl"
        return name
    
    def get_data_loader(self, data_file_name):
        """
            INPUT: data_file_name: str
            OUTPUT: data_type: str
        """
        data_type = data_file_name.split(".")[-1]
        if data_type not in self.supported_data_types:
            print(f"The data type {data_type} is not supported. Please check the supported data types by calling `show_supported_data_types()`.")
            return None
        # A switcher for the data loaders
        self.switcher = {
            "txt": UnstructuredFileLoader,
            "csv": CSVLoader,
            "json": JSONLoader,
            "html": UnstructuredHTMLLoader,
            "md": UnstructuredMarkdownLoader,
            "pdf": UnstructuredPDFLoader
        }
        return self.switcher.get(data_type)

    def ingest_data(self, data_file_name, updata_existing_db=True):
        """
            INPUT: data_file_name: str
            OUTPUT: None

            This function is for ingesting the unstructured data into the database. Unstructured data means that the data is not in the form of a table, like `txt` format.
            Only available for embedding each document individually (different from FAISS approach).
        """
        if not updata_existing_db:
            if self.check_existence(data_file_name):
                print(f"The database {data_file_name} already exists. Please set `update_existing_db=True` to update the database.")
                return
        db_name = self.create_embeddingdb_name(data_file_name)
        print(f"Loading data {data_file_name}...")
        data_path = os.path.join(self.data_dir, data_file_name)
        loader = self.get_data_loader(data_file_name)
        data_loader = loader(data_path)
        data = data_loader.load()

        print(f"Splitting text {data_file_name}...")
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
        )
        documents = text_splitter.split_documents(data)

        print(f"Creating vectorstore {data_file_name}...")
        # set the OpenAI API key
        self.set_OPENAI_API_KEY()

        # create the vectorstore
        embeddings = OpenAIEmbeddings()
        print("=====================================")
        print("Start connecting OpenAI API KEY and costing tokens usage...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        with open(os.path.join(self.db_dir, db_name), "wb") as f:
            pickle.dump(vectorstore, f)
        self.db_list.append(db_name)
        print("=====================================")
        print(f"Successfully created the vectorstore {db_name}.")
    
    def run_data_sync_pkl(self, data_dir=config.get_data_dir(), update=False):
        """
            This function is for running the program of loading the raw data, chunk the data, and store the embeddings.
            Only available for embedding each document individually (different from FAISS approach).
        """
        if len(os.listdir(data_dir)) == 0:
            print(f"The data directory {data_dir} is empty. Please check the data directory.")
            return
        
        for data_file_name in os.listdir(data_dir):
            # Check if the data file is already in the database
            if self.check_existence(data_file_name) and not update:
                print(f"The database {data_file_name} already exists. Please set `update_existing_db=True` to update the database.")
                continue
            else:
                self.ingest_data(data_file_name)
                print(f"Successfully ingested the data {data_file_name}.")
                print("=====================================")

        print("Successfully ingested all the data files.")
        print("=====================================")

    def run_data_sync_faiss(self, data_dir=config.get_update_data_dir(), update=False):
        """
            Create a FAISS vector store from texts, save it locally, load it again, and update it with new documents
        """
        self.set_OPENAI_API_KEY()
        embeddings = OpenAIEmbeddings()

        if self.faiss_exists and not update:
            print("The FAISS vector store already exists. Please set `update=True` to update the vector store.")
            return
        if len(os.listdir(self.update_data_dir)) == 0:
            print(f"The update data directory {self.update_data_dir} is empty. Please check the data directory.")
            return

        for i, file in enumerate(os.listdir(self.update_data_dir)):
            data_file_name = file
            data_path = os.path.join(self.update_data_dir, data_file_name)
            loader = self.get_data_loader(data_file_name)
            data_loader = loader(data_path)
            # initialize data
            data = None
            data = data_loader.load()

            if i == 0:
                faiss = FAISS.from_documents(data, embeddings)
            else:
                faiss.add_documents(data)

            print(f"Successfully load the data {data_file_name} into vectorstore.")
            print(f"Cut the data {data_file_name} from the update data directory into {self.data_dir}...")
            os.rename(os.path.join(self.update_data_dir, data_file_name), os.path.join(self.data_dir, data_file_name))
            print(f"Successfully cut the data {data_file_name} from the update data directory into {self.data_dir}.")
            print("=====================================")

        # Save the vector store locally
        if self.faiss_exists:
            original_faiss = FAISS.load_local(self.vector_dir, embeddings)
            faiss.merge_from(original_faiss)
            faiss.save_local(self.vector_dir)
        else:
            faiss.save_local(self.vector_dir)
        print(f"Successfully saved the vector store into {self.vector_dir}.")
        print("=====================================")


