import os
import re
from pathlib import Path
from ruamel.yaml import YAML
from typing import List, Tuple, Callable

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


def merge_hyphenated_words(_text: str) -> str:
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", _text)


def fix_newlines(_text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", _text)


def remove_multiple_newlines(_text: str) -> str:
    return re.sub(r"\n{2,}", "\n", _text)


def clean_text(_text: str, _cleaning_functions: List[Callable[[str], str]]
               ) -> List[Tuple[int, str]]:

    for cleaning_function in _cleaning_functions:
        _text = cleaning_function(_text)

    return _text


def text_to_docs(_text: str) -> List[Document]:
    doc_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        separators=["\n\n", "\n", ".", "?", "!", " ", ",", ""],
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(_text)

    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "chunk": i,
                "source": f"c{i}",
            },
        )
        doc_chunks.append(doc)

    return doc_chunks


if __name__ == "__main__":
    txt_dir_path = Path("/res/text")
    data_saving_path = Path("C:/Users/geon/PycharmProjects/crafter-llm-actor/data/chroma")
    secrets_path = Path("C:/Users/geon/PycharmProjects/crafter-llm-actor/secrets.yaml")

    assert txt_dir_path.is_dir(), f"txt_dir_path: {txt_dir_path} is not a directory"
    assert data_saving_path.is_dir(), f"data_saving_path: {data_saving_path} is not a directory"
    assert secrets_path.is_file(), f"secrets_path: {secrets_path} is not a file"

    yaml = YAML()
    with open(str(secrets_path), "r") as f:
        secrets = yaml.load(f)

    os.environ["OPENAI_API_KEY"] = secrets["api_key"]["openai"]

    txt_files_path = txt_dir_path.glob("*.text")
    collection_name_list = []

    for txt_file_path in txt_files_path:
        txt_file_path_str = str(txt_file_path)
        print(f"Processing {txt_file_path_str}")

        with open(txt_file_path_str, "r") as f:
            text = f.read()

        cleaning_functions = [
            merge_hyphenated_words,
            fix_newlines,
            remove_multiple_newlines,
        ]

        cleaned_text = clean_text(text, cleaning_functions)
        text_chunks = text_to_docs(cleaned_text)

        embeddings = OpenAIEmbeddings()
        collection_name = txt_file_path.stem.strip().replace(' ', '_')
        collection_name_list.append(collection_name)

        vector_store = Chroma.from_documents(
            text_chunks,
            embeddings,
            collection_name=collection_name,
            persist_directory=str(data_saving_path),
        )

        vector_store.persist()
        print(f"Finished processing {txt_file_path_str}")

    print(f"Finished processing all files in {txt_dir_path}")
    print(f"Data saved to {str(data_saving_path)}")
    print(f"Collection names: {collection_name_list}")
