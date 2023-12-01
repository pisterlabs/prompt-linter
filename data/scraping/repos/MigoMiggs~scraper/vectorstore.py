from config.settings import Config
import logging
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
import langchain
from langchain.document_loaders import PyPDFium2Loader, TextLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk


class VectorDB:
    """
    This class is responsible for loading and managing the vector database
    """

    _instance = None
    vectordb = None
    config = None
    persistance_path: str = ''
    logger = None
    embeddings: langchain.embeddings = None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Constructor
        """

        if not hasattr(self, 'initialized'):
            self.initialized = True

            self.config = Config('../config/config.yaml')
            self.persistance_path = self.config.data['vectordb']['chroma-path']
            self.logger = logging.getLogger('app.Scraper')
            self.embeddings: langchain.embeddings

            self.logger.debug("Load embeddings")

           # load the embeddings model based on the config file setting
            if self.config.get_embeddings()["type"] == 'HUGGING_FACE':

                # load the MiniLM model from Hugging Face
                embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

            elif self.config.get_embeddings()["type"] == 'OPEN_AI':

                # load the Open AI model
                self.embeddings = OpenAIEmbeddings()
            elif self.config.get_embeddings()["type"] == 'HUGGING_FACE_GTE_BASE':

                # load the GTE mode from Hugging Face
                embeddings_model_name = 'thenlper/gte-base'
                self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
            else:

                # unknown embeddings setting
                self.logger.error("Unknown embeddings setting.")
                raise Exception("Unknown embeddings setting.")

            self.logger.debug("Embeddings Loaded")
            self.vectordb = Chroma(persist_directory=self.persistance_path, embedding_function=self.embeddings)

    def is_dense(self, document: Document, min_tokens=300):
        """
        Check if a document is dense enough to be split

        :param document:
        :param min_tokens:
        :return:
        """

        content = document.page_content
        tokens = nltk.word_tokenize(content)
        return len(tokens) > min_tokens

    def split_embed_store(self, filename: str, ext: str) -> None:
        """
        Split, embed and store a file into the vector database
        :param filename:
        :param ext:
        :return:
        """

        self.logger.debug("Split, embed and store " + filename)

        loader = None
        try:
            if ext == '.pdf':
                loader = PyPDFium2Loader(file_path=filename, extract_images=False)
                self.logger.debug('Load PDF')
            else:
                loader = TextLoader(filename)
                self.logger.debug('Load Text')

            if loader is not None:

                # load the documents without split
                docs = loader.load()

                # remove the documents that are not dense
                docs = [doc for doc in docs if self.is_dense(doc)]

                if (len(docs)) == 0:
                    self.logger.info(f'No dense documents in {filename}')
                    return

                # remove from each document lines that are too short
                for doc in docs:
                    lines = doc.page_content.split('\n')
                    lines = [line for line in lines if len(line) > 50]
                    doc.page_content = '\n'.join(lines)

                # split the documents
                splitdocs = self.text_splitter.split_documents(docs)
                self.vectordb.add_documents(documents=splitdocs)

        except Exception as e:
            self.logger.error(f'Failed to split and embed: {filename}')
            self.logger.error(f'Exception: {e}')

    def get_vector_db(self):
        """

        :return:
        """
        return self.vectordb
