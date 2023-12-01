# from datetime import date
from typing import Dict, List, Tuple
from loguru import logger
from langchain.document_loaders import BSHTMLLoader
# local imports
from ingest.ingest_utils import IngestUtils


class HtmlParser:
    """A parser for extracting text from HTML documents."""

    def __init__(self, chunk_size: int, chunk_overlap: int, file_no: int, text_splitter_method: str):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_no = file_no
        self.text_splitter_method = text_splitter_method

    def parse_html(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """Extract and return the pages and metadata from the PDF."""
        ingestutils = IngestUtils(self.chunk_size, self.chunk_overlap, self.file_no, self.text_splitter_method)
        
        # load text and extract raw page 
        logger.info("Extracting pages")
        loader = BSHTMLLoader(file_path, open_encoding='utf-8')
        data = loader.load()
        raw_text = data[0].page_content.replace('\n', '')

        pages = [(1, raw_text)] # html files do not have multiple pages

        # extract metadata
        logger.info("Extracting metadata")
        metadata_text = data[0].metadata
        logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
        metadata = {"title": ingestutils.getattr_or_default(metadata_text, 'title', '').strip(),
                   "author": ingestutils.getattr_or_default(metadata_text, 'author', '').strip(),
                   "filename": file_path.split('\\')[-1]
        }
        return pages, metadata