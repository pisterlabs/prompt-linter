from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter

def text_splitter():
    #text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    #     chunk_size=100,
    #     chunk_overlap=0,
    #     length_function=len,
    #     add_start_index=True,
    # )
    # texts = text_splitter.create_documents([meta_snippet])


def markdown_splitter():
    meta_snippet = ""
    # This is a long document we can split up.
    with open("transcripts/meta.md") as f:
        meta_snippet = f.read()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    texts = md_splitter.split_text(meta_snippet)

    print(texts)
    print(texts[0])
    print(texts[1])
    print(texts[2])

    print(len(texts))
