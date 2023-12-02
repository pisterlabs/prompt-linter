import requests
import re
import math
from bs4 import BeautifulSoup
from typing import List
from numpy import array_split


from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


KEYWORD_CHUNK_SIZE = 10


class RPALoader(BaseLoader):
    """Load RPA Framework documentation

    Args:
        url: URL to the target RPA Framework documentation definitions
        black_list: List of libraries that should be skipped from loading
    """

    def __init__(self, url: str, black_list: List[str]):
        self.url = url
        self.black_list = black_list

    def load(self) -> List[Document]:
        response = requests.get(self.url)

        if response.status_code == 200:
            json_data = response.json()
        else:
            raise Exception(f"Could not fetch RPA Framework libraries")

        output: List[Document] = []

        for lib_name in json_data:
            if lib_name in self.black_list:
                continue

            library = json_data[lib_name]
            document = Document(
                page_content=self._get_lib_content(library),
                metadata={
                    "title": f"Robot Framework {library['name']} library documentation",
                    "source": self._get_lib_source(library["name"]),
                },
            )
            output.append(document)

            keyword_documents = self._get_keyword_docs(library)
            output.extend(keyword_documents)

        return output

    def _get_lib_source(self, library: str) -> str:
        library_slug = re.sub(r"[^a-z0-9]+", "-", library.lower())
        return f"https://robocorp.com/docs/libraries/rpa-framework/{library_slug}"

    def _get_lib_content(self, library) -> str:
        soup = BeautifulSoup(library["doc"], "html.parser")
        lines = soup.get_text().splitlines()
        description = next((item for item in lines if item), None)

        return f"Robot Framework library {library['name']} - {description}"

    def _get_keyword_docs(self, library) -> List[Document]:
        output = []

        if len(library["keywords"]) == 0:
            return output

        chunks = array_split(
            library["keywords"],
            math.ceil(len(library["keywords"]) / KEYWORD_CHUNK_SIZE),
        )

        for keywords in chunks:
            content = f"Library {library['name']} keyword documentation:\n"
            for keyword in keywords:
                keyword_name = keyword["name"]
                keyword_desc = keyword["shortdoc"]
                keyword_args = ", ".join([arg["name"] for arg in keyword["args"]])

                content = f"{content}\n### {keyword_name}\n Description: {keyword_desc}\n Args: {keyword_args}"

            document = Document(
                page_content=content,
                metadata={
                    "title": f"Library {library['name']} keyword documentation",
                    "source": self._get_lib_source(library["name"]) + "/keywords",
                },
            )

            output.append(document)

        return output