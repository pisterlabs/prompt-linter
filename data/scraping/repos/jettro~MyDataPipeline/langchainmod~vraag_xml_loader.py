import xml.etree.ElementTree as ET
from urllib.request import urlopen

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document


class VraagXMLLoader(BaseLoader):
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
            for document in root:
                # Extract relevant data from the XML element
                text = document.find("question").text
                metadata = {"docid": document.find("id").text, "dataurl": document.find("dataurl").text}
                # Create a Document object with the extracted data
                doc = Document(page_content=text, metadata=metadata)
                # Append the Document object to the list of documents
                docs.append(doc)

            return docs
