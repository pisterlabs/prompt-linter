from langchain.document_loaders import WikipediaLoader
from typing import List
from langchain.docstore.document import Document

class WikipediaLoaderWrapper:

    @staticmethod
    def load(
        query: str, 
        load_max_docs: int,
        doc_content_chars_max: int 
    ) -> List[Document]:
        return WikipediaLoader(
            query = query,
            load_max_docs = load_max_docs,
            doc_content_chars_max = doc_content_chars_max
        ).load()