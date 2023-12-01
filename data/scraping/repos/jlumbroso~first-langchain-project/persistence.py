import os
import typing

import langchain.document_loaders
import langchain.text_splitter
import langchain.embeddings
import langchain.vectorstores
import loguru


class Document:
    """Base class representing a generic document."""

    def __init__(self, filepath: str):
        """Initialize the document with its file path."""
        self.filepath = filepath

    @classmethod
    def from_path(cls, filepath: str):
        """Create a specific document instance based on its file extension."""
        extension = os.path.splitext(filepath)[1].lower()
        if extension == ".pdf":
            return PDFDocument(filepath)
        elif extension in [".tex"]:
            return LaTeXDocument(filepath)
        elif extension in [".doc", ".docx"]:
            return WordDocument(filepath)
        else:
            raise NotImplementedError(f"Unsupported file type: {extension}")

    @property
    def text_splitter(self):
        return None

    def load(self) -> str:
        """Load the content of the document. To be implemented by subclasses."""
        raise NotImplementedError


class PDFDocument(Document):
    """Represents a PDF document."""

    def load(self) -> str:
        """Load the content of the PDF document."""
        return langchain.document_loaders.PyPDFLoader(self.filepath).load()


class WordDocument(Document):
    """Represents a Word document."""

    def load(self) -> str:
        """Load the content of the Word document."""
        return langchain.document_loaders.Docx2txtLoader(self.filepath).load()


class LaTeXDocument(Document):
    """Represents a Word document."""

    def load(self) -> str:
        """Load the content of the Word document."""
        return langchain.document_loaders.TextLoader(self.filepath).load()

    @property
    def text_splitter(self):
        return langchain.text_splitter.LatexTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
        )


class Persistence:
    """Handles the persistence and indexing of documents."""

    DEFAULT_INDEX_DIR = "embeddings"

    def __init__(self, index_path: str = DEFAULT_INDEX_DIR):
        """Initialize the persistence instance."""
        self.index_path = index_path
        self.input_paths = []
        self.text_splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
        )
        self.all_splits = []
        self.all_documents = set()
        self.storage = None  # this can hold the Chroma object or similar storage system

        self.reloaded = None
        if os.path.exists(self.index_path):
            self.reloaded = Persistence.__reload_index(self.index_path)

    @staticmethod
    def __reload_index(index_path: str):
        """Reload an existing index from the specified directory."""
        return langchain.vectorstores.Chroma(
            persist_directory=index_path,
            embedding_function=langchain.embeddings.OpenAIEmbeddings(),
        )

    def __rebuild_storage(self):
        self.storage = langchain.vectorstores.Chroma.from_documents(
            documents=self.all_splits,
            embedding=langchain.embeddings.OpenAIEmbeddings(),
            persist_directory=self.index_path or self.DEFAULT_INDEX_DIR,
        )

        if self.reloaded:
            reloaded_data = self.reloaded._collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            self.storage._collection.add(
                embeddings=reloaded_data.get("embeddings"),
                metadatas=reloaded_data.get("metadatas"),
                documents=reloaded_data.get("documents"),
                ids=reloaded_data.get("ids"),
            )

    def as_retriever(self, *args, **kwargs):
        """Return a retriever object based on the storage."""
        if self.storage:
            return self.storage.as_retriever(*args, **kwargs)

    def index_documents_from_dirs(self, input_paths: typing.List[str]):
        """Index the documents in the specified input directories."""
        data_added = 0
        for input_path in input_paths:
            data_added += self.index_documents_from_dir(input_path, no_rebuild=True)

        # rebuild storage
        self.__rebuild_storage()

        return data_added

    def index_documents_from_dir(self, input_path: str, no_rebuild: bool = False):
        """Index the documents in the specified input directory."""

        data_added = 0
        loguru.logger.debug(f"Indexing documents from: {input_path}")

        for filename in os.listdir(input_path):
            filepath = os.path.join(input_path, filename)

            if filepath in self.all_documents:
                loguru.logger.debug(f"Skipping already indexed file: {filepath}")
                continue

            loguru.logger.debug(f"Indexing file from: {filepath}")

            try:
                document = Document.from_path(filepath)
            except NotImplementedError:
                loguru.logger.info(f"Unsupported file type: {filepath}")
                continue
            except Exception as e:
                loguru.logger.error(f"Unexpected error loading file: {filepath}")
                loguru.logger.error(e)
                continue

            # ensure we can load it
            data = document.load()

            # select splitter
            text_splitter = document.text_splitter or self.text_splitter
            loguru.logger.info(f"Using text splitter: {text_splitter}")

            # record the document
            self.all_documents.add(filepath)
            self.all_splits += text_splitter.split_documents(data)

            # save statistics
            data_added += len(data)

        # rebuild storage
        if not no_rebuild:
            self.__rebuild_storage()

        return data_added

    @classmethod
    def build_from_dirs(
        cls, *input_paths: typing.List[str], index_path: str = DEFAULT_INDEX_DIR
    ):
        """Retrieve the storage based on the input directory."""
        instance = cls(index_path=index_path)
        instance.index_documents_from_dirs(input_paths=input_paths)
        return instance


# Minimum usage example:
"""
# Fetch the storage for a given input directory
persistence_instance = Persistence.build_from_dirs("/path/to/documents")

# Access the storage or index as needed.
storage = persistence_instance.as_retriever()
"""
