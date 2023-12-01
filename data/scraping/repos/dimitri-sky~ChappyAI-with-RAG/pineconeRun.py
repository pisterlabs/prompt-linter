# fixes a bug with asyncio and jupyter
import nest_asyncio
nest_asyncio.apply()

from langchain.document_loaders.sitemap import SitemapLoader

loader = SitemapLoader(
    "https://www.cod.edu/sitemap.xml",
    filter_urls=["https://www.cod.edu/about/"]
)
docs = loader.load()
print("\nDocs", docs)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap  = 200,
    length_function = len,
)

docs_chunks = text_splitter.split_documents(docs)