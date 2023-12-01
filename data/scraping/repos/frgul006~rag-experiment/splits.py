import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm

from regent_rag.core.logging import logger
from regent_rag.core.settings import get_settings

tokenizer = tiktoken.get_encoding("cl100k_base")


def tiktoken_len(text: str) -> int:
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""],
)


def process_file(file_path: str) -> list[Any]:
    documents = []

    try:
        # Load the JSON content
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        # Generate a unique ID based on the file path
        m = hashlib.md5()
        m.update(file_path.encode("utf-8"))
        uid = m.hexdigest()[:12]

        # Split the content into chunks
        chunks = text_splitter.split_text(content["content"])

        # Create document data
        for i, chunk in enumerate(chunks):
            documents.append({"id": f"{uid}-{i}", "text": chunk, "source": content["url"]})

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
    except KeyError as e:
        logger.error(f"Missing key {e} in file: {file_path}")

    return documents


def process_json_files(folder_path: str, output_folder_path: str) -> list[Any]:
    # Get all files in the folder
    all_files = os.listdir(folder_path)

    # Filter out JSON files
    json_files = [file for file in all_files if file.endswith(".json")]

    documents = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(process_file, os.path.join(folder_path, file)): file for file in json_files}
        for future in tqdm(as_completed(future_to_file), total=len(json_files)):
            documents.extend(future.result())

    # Save the documents to a JSONL file
    with open(f"{output_folder_path}/train.jsonl", "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")

    return documents


def main() -> None:
    output_folder = get_settings().output_folder
    scrape_folder = f"{output_folder}/scrape"
    process_json_files(scrape_folder, output_folder)


if __name__ == "__main__":
    main()
