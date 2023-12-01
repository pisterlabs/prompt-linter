# %%
from tasks.process_apify_scrape import process_apify_scrape
from langchain.document_loaders.dataframe import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.storage.file_system import LocalFileStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings


# %%
processed = process_apify_scrape("data/scrape/apify/newstead.json")
processed.to_json(
    "data/schools/www.newsteadwood.co.uk/scrape.json", orient="records", lines=True)


# %%
loader = DataFrameLoader(processed, page_content_column="text")
docs = loader.load()
print(f"Doc loaded {len(docs)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, separators=['\n\n', '\n', ' '])
chunked_docs = text_splitter.split_documents(docs)
print(f"Chunked docs {len(chunked_docs)}")
underlying_embeddings = OpenAIEmbeddings(chunk_size=20)
fs = LocalFileStore("./.cache/")


cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, fs, namespace=underlying_embeddings.model
)

db = Chroma.from_documents(
    chunked_docs, cached_embedder, persist_directory="data/embeddings/multi")
