import os
import time
import datetime
import warnings
# from etl.utils.config import ACTION

from remove_jupyter_text import RemoveJupyterText

import luigi

import openai

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

import weaviate
from weaviate.util import generate_uuid5

from tqdm.auto import tqdm
from bs4 import GuessedAtParserWarning

from utils.logger import Logger
from utils.config import ACTION, DATA
from utils.weaviate_config import WEAVIATE_CLASS, WEAVIATE_SCHEMA

# Initialize logger
logger = Logger()

class Upsert(luigi.Task):
    """
    Task that performs upsert operations on a Weaviate Vectorstore Instance.

    This task performs the following operations:
    1. Load text documents from a directory.
    2. Split the documents into chunks.
    3. Perform an upsert operation on a Weaviate instance with the chunks.

    This task makes use of several external libraries including 'luigi' for
    workflow management, 'weaviate' for interaction with the Weaviate Vectorstore
    Instance, and 'tiktoken' for text tokenization.

    Attributes:
    directory: A luigi.Parameter instance representing the directory to load text documents from.
    date_time: A luigi.Parameter instance representing the current date and time.

    """

    default_directory = DATA['LAKE_EXTRAS']
    default_date_time = datetime.datetime.now().strftime(DATA["DATETIME"])

    directory = luigi.Parameter(default=default_directory)
    date_time = luigi.Parameter(default=default_date_time)

    def requires(self):
        """
        Specifies the dependency of this task. This task requires the completion
        of the RemoveJupyterText task.

        Returns:
        An instance of the RemoveJupyterText task.
        """
        return RemoveJupyterText()

    # The main function to run the Task
    def run(self):
        with self.output().open('w') as out_file:
            logger.ok(ACTION['START'], f"Build Data Lake")
            """
            The main method that runs when the Task is executed.
            It orchestrates the execution of other methods in the proper sequence.
            """
            try:
                self.initialize_variables()
                self.client = self.create_weaviate_client()
                self.tokenizer = self.configure_tokenizer()
                self.text_splitter = self.initialize_text_splitter()

                docs = self.load_docs()
                documents = self.split_docs_into_chunks(docs)

                self.manage_weaviate_schema()

                self.batch_upsert(documents)
            except Exception as e:
                logger.error(f"An error occurred while running the Luigi Task.", str(e))
                raise

    def initialize_variables(self):
        """
        Initializes several necessary variables, such as the OpenAI API key and the Weaviate host.
        It also sets the OpenAI API key in the openai library.
        Raises an EnvironmentError if the necessary environment variables are not set.
        """
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")
        self.WEAVIATE_AUTH_API_KEY = weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_AUTH_API_KEY")) # BEARER_TOKEN

        if not self.OPENAI_API_KEY or not self.WEAVIATE_HOST or not self.WEAVIATE_AUTH_API_KEY:
            logger.error("Environment variables are not set properly.")
            raise EnvironmentError("Environment variables are not set properly.")

        openai.api_key = self.OPENAI_API_KEY

        # Ignore specific warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

    def create_weaviate_client(self):
        """
        Creates and returns a Weaviate client using the weaviate library.
        Raises an exception if an error occurs while creating the client.
        """
        try:
            return weaviate.Client(
                url=self.WEAVIATE_HOST,
                auth_client_secret=self.WEAVIATE_AUTH_API_KEY,
                additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
            )
        except Exception as e:
            logger.error(f"An error occurred while creating Weaviate client.", str(e))
            raise

    def configure_tokenizer(self):
        """
        Configures and returns a tokenizer using the tiktoken library.
        Raises an exception if an error occurs while configuring the tokenizer.
        """
        try:
            return tiktoken.get_encoding('cl100k_base')
        except Exception as e:
            logger.error(f"An error occurred while configuring the tokenizer.", str(e))
            raise

    def initialize_text_splitter(self):
        """
        Initializes and returns a text splitter using the RecursiveCharacterTextSplitter class from the langchain library.
        Raises an exception if an error occurs while initializing the text splitter.
        """
        try:
            return RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=20,
                length_function=self.tiktoken_len,
                separators=['\\n\\n', '\\n', ' ', '']
            )
        except Exception as e:
            logger.error(f"An error occurred while initializing the text splitter.", str(e))
            raise

    def tiktoken_len(self, text):
        """
        Calculates and returns the length of a token using the configured tokenizer.
        Raises an exception if an error occurs while calculating the token length.
        """
        try:
            tokens = self.tokenizer.encode(
                text,
                disallowed_special=()
            )
            return len(tokens)
        except Exception as e:
            logger.error(f"An error occurred while calculating token length.", str(e))
            raise

    def load_docs(self):
        """
        Loads and returns documents from the specified directory using the DirectoryLoader class from the langchain library.
        Raises an exception if an error occurs while loading the documents.
        """
        try:
            logger.walk("DirectoryLoader", f"Started  - {self.directory}")
            loader = DirectoryLoader(path=self.directory, glob="**/*.md", recursive=True, show_progress=True)
            logger.walk("DirectoryLoader",f"Ended  - {self.directory}")

            logger.walk("loader.load()", "Started")
            load = loader.load()
            logger.walk("loader.load()", "Ended")
            return load

        except Exception as e:
            logger.error(f"An error occurred while loading documents.", str(e))
            raise

    def split_docs_into_chunks(self, docs):
        """
        Splits the documents into chunks using the initialized text splitter.
        Returns a list of dictionaries, where each dictionary represents a chunk of text.
        Raises an exception if an error occurs while splitting the documents.
        """
        documents = []
        total_tokens_len = 0
        min_chunk_len = float('inf')
        max_chunk_len = float('-inf')
        for doc in tqdm(docs):
            try:
                chunks = self.text_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(chunks):
                    chunk_len = self.tiktoken_len(chunk)
                    # document_id = generate_uuid5(chunk)
                    documents.append({
                        'content': chunk,
                        # 'document_id': document_id,
                    })
                    total_tokens_len += chunk_len
                    min_chunk_len = min(min_chunk_len, chunk_len)
                    max_chunk_len = max(max_chunk_len, chunk_len)
                    logger.walk(f"Chunk Length:", chunk_len)
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"An error occurred while splitting document into chunks.", str(e))
                raise
        # log the total tokens
        logger.ok("Total Tokens:", total_tokens_len)
        logger.ok("Minimum Chunk Length:", min_chunk_len)
        logger.ok("Maximum Chunk Length:", max_chunk_len)
        return documents

    def manage_weaviate_schema(self):
        """
        Manages the schema in Weaviate, including deleting and creating a class.
        Raises an exception if an error occurs while managing the schema.
        """
        try:
            # Delete a schema class from Weaviate. This deletes all associated data
            self.client.schema.delete_class(WEAVIATE_CLASS)
            logger.ok(f"Weaviate Class Deleted", WEAVIATE_CLASS)
        except Exception as e:
            logger.error(f"An error occurred while deleting a Weaviate schema", str(e))
            raise
        try:
            # Create the schema of the Weaviate instance, with all classes at once
            self.client.schema.create(WEAVIATE_SCHEMA)
            logger.ok(f"Weaviate Class Created", WEAVIATE_SCHEMA)
        except Exception as e:
            logger.error(f"An error occurred while creating a Weaviate schema", str(e))
            raise

    def batch_upsert(self, documents):
        """
        Inserts documents in batch to Weaviate using the batch functionality of the Weaviate client.
        Raises an exception if an error occurs while inserting the documents.
        """
        with self.client.batch as batch:
            for document in documents:
                try:
                    data_object = {
                        "content": str(document["content"]),
                        # "document_id": str(document["document_id"])
                    }
                    uuid = generate_uuid5(data_object)
                    # logger.walk(f"str(document['document_id']):", data_object["document_id"])

                    # https://weaviate.io/developers/weaviate/manage-data/import
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=WEAVIATE_CLASS,
                        uuid=uuid,
                        # vector: Optional[Sequence] = None,
                    )
                    logger.ok(f"Upsert Chunk UUID:", uuid)
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"An error occurred while upserting document", str(e))
                    raise

    def output(self):
        """
        Defines the output of the Task.
        In this case, the output is a log file.
        """
        return luigi.LocalTarget(f'{DATA["LOGS"]}/{self.date_time}_upsert.log')

if __name__ == "__main__":
    """
    Main block that builds and executes the Upsert task using Luigi's build function.
    """
    try:
        luigi.build([Upsert()], local_scheduler=False)
    except Exception as e:
        logger.error(f"An error occurred while building the Task.", str(e))
        raise
