import openai
import pathlib
import sys
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.document_loaders import PyPDFLoader

_parentdir = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
print(_parentdir)

from scripts.config import Config

cfg = Config()

openai.api_key = cfg.openai_api_key

loader = PyPDFLoader("examples/docs/api.pdf")
pages = loader.load()

r_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150, separators=["", " ", "\n", "\n\n", "(?<=\. )"]
)

c_text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separator="\n",
    length_function=len,
)

c_docs = c_text_splitter.split_documents(pages)
r_docs = r_text_splitter.split_documents(pages)

print(c_docs[45])
print(r_docs[45])
