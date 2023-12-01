from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os


from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("2103.15348.pdf")
pages = loader.load_and_split()

# Classic
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

texts = text_splitter.split_documents(pages)

# print(texts[0])

print(f"you have {len(texts)} documents")
print("Preview: ", texts[0].page_content)

# Recursive Character Text Splitter

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    length_function=len,
)

docs = text_splitter.split_documents(pages)
for doc in docs:
    print(doc)


# NTLK Text Splitter
from langchain.text_splitter import NLTKTextSplitter
text_splitter = NLTKTextSplitter(chunk_size=500)
texts = text_splitter.split_text(texts[0].page_content)
print(texts)

# Spacy Text Splitter
from langchain.text_splitter import SpacyTextSplitter
text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_text(texts[0].page_content)
print(texts)

# Token Text Splitter
from langchain.text_splitter import TokenTextSplitter
text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_text(texts[0].page_content)
print(texts)