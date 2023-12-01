# HOW TO SPLIT DOCUMENTS into chunks SINCE THEY SHOULD BE BIG

# package langchain.textsplitter
# doc https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/character_text_splitter

# textsplitters in langchain have all these methods
# - create_documents()  create documents from a list of text
# - split_documents() create split documents

# IMPORTANT NOTE
# Documents loaded with document loaders are the same type of Documents created with splitters
# At the end , they are all Documents

import os, sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


def document_splitting():
    chunk_size = 26
    chunk_overlap = 4

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    c_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # 26 char so the chunk will be only 1
    # text1 = "abcdefghijklmnopqrstuvwxyz"
    # r_splitted = r_splitter.split_text(text1)
    # print(r_splitted)

    # # more than 26 char so the chunk will be more than 1
    # text2 = "abcdefghijklmnopqrstuvwxyzabcdefg"
    # r_splitted = r_splitter.split_text(text2)
    # print(r_splitted)

    # case with spaces
    text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"

    # r_splitted = r_splitter.split_text(text3)
    # print(r_splitted)

    # c_splitted = c_splitter.split_text(text3)
    # print(c_splitted)
    # print only one chunk because the default separator is the NEW LINE
    # ['a b c d e f g h i j k l m n o p q r s t u v w x y z']

    # change the separator to SPACE
    c_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" "
    )
    c_splitted = c_splitter.split_text(text3)
    print(c_splitted)


# RecursiveCharacterTextSplitter` is recommended for generic text.
def recoursive_splitting():
    some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

    print(len(some_text))

    # c_splitter = CharacterTextSplitter(chunk_size=450, chunk_overlap=0, separator=" ")

    # r_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=450, chunk_overlap=0, separators=["\n\n", "\n", " ", ""]
    # )

    # c_splitted = c_splitter.split_text(some_text)
    # print(c_splitted)

    # r_splitted = r_splitter.split_text(some_text)
    # print(r_splitted)

    # Let's reduce the chunk size a bit and add a period to our separators:
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150, chunk_overlap=0, separators=["\n\n", "\n", "\. ", " ", ""]
    )
    r_splitted = r_splitter.split_text(some_text)
    print(r_splitted)


def pdf_splitting():
    # load a pdf using the document loader pypdf loader
    from langchain.document_loaders import PyPDFLoader

    loader = PyPDFLoader("./documents/react paper.pdf")
    pages = loader.load()

    from langchain.text_splitter import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=150, length_function=len
    )

    # use split document since we are working with document and not with simple text
    docs = text_splitter.split_documents(pages)
    print("Doc information")
    print(len(docs))
    print(docs[0].page_content[:2000])
    print("************")

    print("Page information")
    print(len(pages))
    print(pages[0].page_content[:2000])
    print("************")


def token_splitting():
    from langchain.text_splitter import TokenTextSplitter
    from langchain.document_loaders import PyPDFLoader

    # text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
    # text1 = "foo bar bazzyfoo"

    # text_splitted = text_splitter.split_text(text1)
    # print(text_splitted)

    # load pdf for testing with documents
    loader = PyPDFLoader("./documents/react paper.pdf")
    pages = loader.load()

    text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

    docs = text_splitter.split_documents(pages)

    print(docs[1])
    print(docs[1].metadata)

    print(len(docs))
    print(len(pages))


def markdown_splitting():
    from langchain.text_splitter import MarkdownHeaderTextSplitter

    markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n 
## Chapter 2\n\n \
Hi this is Molly"""

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    md_header_splits = markdown_splitter.split_text(markdown_document)

    print(md_header_splits[0])
    print(md_header_splits[1])


token_splitting()


# NOTE
# I can add custom metadata to the chunked documents using this code
#     cont = 1
#     for doc in docs:
#         doc.metadata["prova"] = f"prova_{cont}"
#         cont += 1

#     print(docs[3])
