import logging
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
import langchain

class KB:
    _db: Chroma
    _embedding: langchain.embeddings
    _path: str
    _name: str
    _embedding_type: str

    def __init__(self, name, directory, embeddings_type):
        logger = logging.getLogger('server')
        logger.debug(f'Creating Knowledge Base: {name}')

        self._name = name
        self._embeddings_type = embeddings_type
        self._directory = directory

        # Instantiate the right embeddings
        if embeddings_type == 'HUGGING_FACE':
            logger.debug(f'Creating Hugging Face Embeddings')
            embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self._embedding = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        elif embeddings_type == 'OPEN_AI':
            logger.debug(f'Creating OpenAI Embeddings')
            self._embedding = OpenAIEmbeddings()
        else:
            logger.error("Unknown embeddings setting.")
            raise Exception("Unknown embeddings setting.")

        # create the db
        self._db = Chroma(persist_directory=directory, embedding_function=self._embedding)
        logger.debug(f'Created {name}')

    def get_db(self):
        return self._db

    def get_embeddings(self):
        return self._embedding

