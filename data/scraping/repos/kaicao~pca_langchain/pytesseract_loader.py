import logging
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


class PytesseractLoader(BaseLoader):
    """Apply OCR on image.


    Args:
        file_path: Path to the file to load.
    """

    def __init__(
        self,
        file_path: str
    ):
        """Initialize with file path."""
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load from file path."""
        text = ''
        try:
            with Image.open(self.file_path) as image:
                text = pytesseract.image_to_string(image)
                print('============================================== OCR RESULT START ================================================')
                print('\n')
                print('OCR result: ' + text)
                print('============================================== OCR RESULT START ================================================')
                print('\n')
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
