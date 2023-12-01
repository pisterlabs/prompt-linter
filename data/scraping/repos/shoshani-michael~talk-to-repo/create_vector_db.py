import json
import os
import subprocess
import copy
from pathlib import Path
from langchain.docstore.document import Document

import pandas as pd
import pinecone
import tiktoken
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

load_dotenv()

embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_organization=os.environ["OPENAI_ORG_ID"],
)
encoder = tiktoken.get_encoding("cl100k_base")

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["ENVIRONMENT"]
)
vector_store = Pinecone(
    index=pinecone.Index(os.environ["PINECONE_INDEX"]),
    embedding_function=embeddings.embed_query,
    text_key="text",
    namespace=os.environ["NAMESPACE"],
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(os.environ["CHUNK_SIZE"]),
    chunk_overlap=int(os.environ["CHUNK_OVERLAP"]),
)


def clone_from_github(REPO_URL, LOCAL_REPO_PATH):
    temp_dir = LOCAL_REPO_PATH
    repo_url = REPO_URL

    # if the repo is already cloned, there's a .git/ folder, do git pull
    if os.path.isdir(os.path.join(temp_dir, ".git")):
        print(f"Pulling {repo_url} into {temp_dir}")
        subprocess.run(["git", "pull"], cwd=temp_dir, check=True)
    else:
        # otherwise, clone the repo
        print(f"Cloning {repo_url} into {temp_dir}")
        subprocess.run(["git", "clone", repo_url, temp_dir], check=True)

    return


def is_unwanted_file(file_name, unwanted_files, unwanted_extensions):
    if (
        file_name.endswith("/")
        or any(f in file_name for f in unwanted_files)
        or any(file_name.endswith(ext) for ext in unwanted_extensions)
    ):
        return True
    return False


def process_file(file):
    file_contents = str(file.read())
    n_tokens = len(encoder.encode(file_contents))
    return file_contents, n_tokens


def file_contains_secrets(filename):
    return False #TODO-M:delete this line
    # Run the detect-secrets command on the temp file
    result = subprocess.run(
        ["python", "-m", "detect_secrets", "scan", filename], capture_output=True
    )
    output = json.loads(result.stdout)

    # Check if any secrets were detected in the file
    return len(output["results"].get(filename, [])) > 0

def index_to_coordinates(s, index):
    """Returns (line_number, col) of `index` in `s`."""
    if not len(s):
        return 1, 1
    sp = s[:index+1].splitlines(keepends=True)
    return len(sp), len(sp[-1])

def create_documents_with_met(splitter, texts, metadatas = None):
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in splitter.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
#                 if self._add_start_index:
                index = text.find(chunk, index + 1)
                start_coordinates = index_to_coordinates(text, index)
                end_coordinates = index_to_coordinates(text, index + len(chunk))
                metadata["start_index"] = index
                metadata["start_line"] = str(start_coordinates[0])
                metadata["start_position"] = str(start_coordinates[1])
                metadata["end_line"] = str(end_coordinates[0])
                metadata["end_position"] = str(end_coordinates[1])
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

def index_to_coordinates(s, index):
    """Returns (line_number, col) of `index` in `s`."""
    if not len(s):
        return 1, 1
    sp = s[:index+1].splitlines(keepends=True)
    return len(sp), len(sp[-1])

def create_documents_with_met(splitter, texts, metadatas = None):
    """Create documents from a list of texts."""
    _metadatas = metadatas or [{}] * len(texts)
    documents = []
    for i, text in enumerate(texts):
        index = -1
        for chunk in splitter.split_text(text):
            metadata = copy.deepcopy(_metadatas[i])
            #                 if self._add_start_index:
            index = text.find(chunk, index + 1)
            start_coordinates = index_to_coordinates(text, index)
            end_coordinates = index_to_coordinates(text, index + len(chunk))
            metadata["start_index"] = index
            metadata["start_line"] = str(start_coordinates[0])
            metadata["start_position"] = str(start_coordinates[1])
            metadata["end_line"] = str(end_coordinates[0])
            metadata["end_position"] = str(end_coordinates[1])
            new_doc = Document(page_content=chunk, metadata=metadata)
            documents.append(new_doc)
    return documents

def process_file_list(temp_dir):
    with open(".db-ignore-files.txt", "r") as f:
        unwanted_files = tuple(f.read().strip().splitlines())
    with open(".db-ignore-extensions.txt", "r") as f:
        unwanted_extensions = tuple(f.read().strip().splitlines())

    corpus_summary = []
    file_texts, metadatas = [], []

    for root, _, files in os.walk(temp_dir):
        for filename in files:
            if not is_unwanted_file(filename, unwanted_files, unwanted_extensions):
                file_path = os.path.join(root, filename)
                if ".git" in file_path:
                    continue
                with open(file_path, "r") as file:
                    if file_contains_secrets(file_path):
                        os.remove(file_path)
                        print(f"Deleted {file_path} as it contained secrets")
                        continue

                    print(f"Processing {file_path}")
                    try:
                        file_contents = file.read()
                    except UnicodeDecodeError:
                        print(f"Skipping {file_path} as it could not be decoded")
                        continue
                    
                    n_tokens = len(encoder.encode(file_contents))
                    file_path = file_path.replace(temp_dir, "").lstrip("/")
                    corpus_summary.append(
                        {"file_name": file_path, "n_tokens": n_tokens}
                    )
                    file_texts.append(file_contents)
                    metadatas.append({"document_id": file_path })

    split_documents = create_documents_with_met(splitter , file_texts, metadatas=metadatas)

    print(f"Writing {len(split_documents)} documents to Pinecone")
    print(os.environ["PINECONE_INDEX"] + " " + os.environ["NAMESPACE"])
    vector_store.from_documents(
        documents=split_documents,
        embedding=embeddings,
        index_name=os.environ["PINECONE_INDEX"],
        namespace=os.environ["NAMESPACE"],
    )

    Path("data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(corpus_summary).to_csv(
        "data/corpus_summary.csv", index=False
    )


def create_vector_db(REPO_URL, LOCAL_REPO_PATH):
    clone_from_github(REPO_URL, LOCAL_REPO_PATH)
    process_file_list(LOCAL_REPO_PATH)
    return


if __name__ == "__main__":
    create_vector_db(os.environ["REPO_URL"], os.environ["LOCAL_REPO_PATH"])
