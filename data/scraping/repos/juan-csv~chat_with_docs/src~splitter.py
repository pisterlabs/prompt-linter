"""Retriever and vector store"""
if True:
    import sys
    sys.path.append("../")
from src.utils.logger import Logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFMinerLoader
from src.utils.config import load_config
# set logger
logger = Logger(__name__).get_logger()


class SplitterException(Exception):
    """Custom class for Splitter Exception Handling"""


class Splitter:
    """Splitter class"""

    def __init__(self, debug=False):
        """Retriever class"""
        try:
            self.debug = debug

            # Load config
            self.config = load_config(debug=self.debug)

            # parameters
            self.chunk_size = self.config['splitter']['chunk_size']
            self.chunk_overlap = self.config['splitter']['chunk_overlap']

            # Instance
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        except Exception as error:
            logger.error(f"Error initializing Splitter class: {error}")
            raise SplitterException(
                f"Exception caught in Splitter module - init: {error}" 
            )

    def process_document(self, path_file: str):
        """Read document -> split in chunks -> return documents list"""
        # Read document
        try:
            doc = PDFMinerLoader(path_file).load()
            if self.debug:
                chunks = self.splitter.split_documents(doc)[:5]
            else:
                chunks = self.splitter.split_documents(doc)
            return chunks
        except Exception as error:
            logger.error(f"Error processing document: {error}")
            raise SplitterException(
                f"Exception caught in Splitter module - process_document: {error}"
            )


if __name__ == "__main__":
    path_file = "../docs_example/contract.pdf"
    splitter = Splitter(debug=True)
    splitted_docs = splitter.process_document(path_file=path_file)
    print('Chunks: ', len(splitted_docs))
    if len(splitted_docs) > 0:
        print(splitted_docs[0].page_content)
