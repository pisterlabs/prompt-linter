import xml.etree.ElementTree as ET
from typing import Optional
from urllib.request import urlopen
from bs4 import BeautifulSoup

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document


class AntwoordXMLLoader(BaseLoader):
    """
    Mainly copied from the GitHub issue:
    https://github.com/hwchase17/langchain/issues/4859
    """

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        super().__init__()
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> list[Document]:
        with urlopen(self.file_path) as f:
            tree = ET.parse(f)
            root = tree.getroot()

            docs = []
            content_blocks = root.findall('.//contentblock')

            # Extract the content of each contentblock element
            content_list = []
            for content_block in content_blocks:
                title = self.__extract_text(content_block.find('paragraphtitle'))
                paragraph = self.__extract_text(content_block.find('paragraph'))
                paragraph = BeautifulSoup(paragraph, 'html.parser').get_text()
                content_list.append(f"{title}\n{paragraph}\n\n")

            # Create a single string with the formatted content
            formatted_content = ''.join(content_list)

            # Create a Document object with the extracted data
            doc = Document(page_content=formatted_content)
            # Append the Document object to the list of documents
            docs.append(doc)

        return docs

    @staticmethod
    def __extract_text(element: Optional[ET.Element]):
        if element is not None:
            return element.text.strip()

        return ""
