from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
    )

    docs = splitter.create_documents([text])
    return docs
