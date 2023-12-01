from llama_index import VectorStoreIndex, download_loader, StorageContext
from llama_index.vector_stores import PineconeVectorStore

"""Simple reader that reads wikipedia."""
from typing import Any, List

from llama_index.readers.base import BaseReader
from llama_index.schema import Document

from dotenv import load_dotenv
import os
import openai
import pinecone

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

class JaWikipediaReader(BaseReader):
    """Wikipedia reader.

    Reads a page.

    """

    def __init__(self) -> None:
        """Initialize with parameters."""
        try:
            import wikipedia  # noqa: F401
        except ImportError:
            raise ImportError(
                "`wikipedia` package not found, please run `pip install wikipedia`"
            )

    def load_data(self, pages: List[str], **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory.

        Args:
            pages (List[str]): List of pages to read.

        """
        import wikipedia
        wikipedia.set_lang("ja")
        results = []
        for page in pages:
            page_content = wikipedia.page(page, **load_kwargs).content
            results.append(Document(text=page_content))
        return results

WikipediaReader = download_loader("WikipediaReader")
loader = JaWikipediaReader()
documents = loader.load_data(pages=['ONE_PIECE', 'ONE_PIECEの登場人物一覧', 'ONE_PIECEの用語一覧', 'ONE_PIECEの地理'])

# init pinecone
pinecone.init(api_key=os.environ["OPENAI_API_KEY"], environment="asia-southeast1-gcp-free")
# pinecone.create_index("manga-reader", dimension=1536, metric="cosine", pod_type="p1")

# construct vector store and customize storage context
storage_context = StorageContext.from_defaults(
    vector_store = PineconeVectorStore(pinecone.Index("manga-reader"))
)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

