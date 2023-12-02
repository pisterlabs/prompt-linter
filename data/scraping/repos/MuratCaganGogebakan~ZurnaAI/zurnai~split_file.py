from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

sol_splitter = RecursiveCharacterTextSplitter.from_language(
    Language.SOL, chunk_size=200, chunk_overlap=0
)


def split_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    docs = sol_splitter.create_documents([text])
    docs = [doc.page_content for doc in docs]
    return text, docs
