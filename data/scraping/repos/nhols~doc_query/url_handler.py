from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from doc_query.app_config import config


class URLHandler:
    def __init__(self, url: str, doc_name: str):
        self.url = url
        self.doc_name = doc_name
        self.text = self._load_text_from_url()

    def _load_text_from_url(self) -> str:
        response = requests.get(self.url)
        if not response.ok:
            raise Exception(f"Could not load url {self.url}")
        return "\n".join(BeautifulSoup(response.content, "html.parser").stripped_strings)

    def split_text(self) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_text(self.text)

    def split_embed_text(self) -> None:
        split_texts = self.split_text()
        config.vectorstore.add_texts(
            texts=[split_text for split_text in split_texts],
            namespace=self.doc_name,
        )
