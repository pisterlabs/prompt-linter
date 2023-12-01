import re
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter


def markdown_to_documents(input: str) -> List[Document]:
    # Remove images
    image_pattern = r"!\[.*?\]\((.*?)\)"
    input = re.sub(image_pattern, "", input)

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    documents = text_splitter.create_documents([input])

    return documents