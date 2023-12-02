import openai
import pathlib
import sys
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
)
from langchain.document_loaders import TextLoader

_parentdir = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
print(_parentdir)

from scripts.config import Config

cfg = Config()

openai.api_key = cfg.openai_api_key

loader = TextLoader("examples/docs/api_introduccion.md")
markdown_document = loader.load()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_split = markdown_splitter.split_text(markdown_document[0].page_content)

print(md_header_split[6])
