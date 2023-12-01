

import os
import json
import copy
import pinecone
import tiktoken
import subprocess
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = OpenAIEmbeddings(openai_api_key = os.environ["OPENAI_API_KEY"], openai_organization = os.environ["OPENAI_ORG_ID"], )
encoder = tiktoken.get_encoding("cl100k_base")

def clone_from_github(REPO_URL, LOCAL_REPO_PATH):
    temp_dir = LOCAL_REPO_PATH
    repo_url = REPO_URL
    if os.path.isdir(os.path.join(temp_dir, ".git")):
        print(f"Pulling {repo_url} into {temp_dir}")
        subprocess.run(["git", "pull"], cwd = temp_dir, check = True)
    else:
        print(f"Cloning {repo_url} into {temp_dir}")
        subprocess.run(["git", "clone", "--depth", "4", repo_url, temp_dir], check = True)
    return

def is_unwanted_file(file_name, unwanted_files, unwanted_extensions):
    if (file_name.endswith("/") or any(f in file_name for f in unwanted_files) or any(file_name.endswith(ext) for ext in unwanted_extensions)): return True
    return False

def file_contains_secrets(filename): return len(json.loads(subprocess.run(["python", "-m", "detect_secrets", "scan", filename], capture_output = True).stdout)["results"].get(filename, [])) > 0

def index_to_coordinates(s, index):
    if not len(s): return 1, 1
    sp = s[:index + 1].splitlines(keepends = True)
    return len(sp), len(sp[-1])

def create_documents_with_met(splitter, texts, metadatas = None):
    _metadatas = metadatas or [{}] * len(texts)
    documents = []
    for i, text in enumerate(texts):
        index = -1
        for chunk in splitter.split_text(text):
            metadata = copy.deepcopy(_metadatas[i])
            index = text.find(chunk, index + 1)
            start_coordinates = index_to_coordinates(text, index)
            end_coordinates = index_to_coordinates(text, index + len(chunk))
            metadata["start_index"] = index
            metadata["start_line"] = str(start_coordinates[0])
            metadata["start_position"] = str(start_coordinates[1])
            metadata["end_line"] = str(end_coordinates[0])
            metadata["end_position"] = str(end_coordinates[1])
            new_doc = Document(page_content = chunk, metadata = metadata)
            documents.append(new_doc)
    return documents

def process_file_list(temp_dir):
    with open(".db-ignore-files.txt", "r") as f: unwanted_files = tuple(f.read().strip().splitlines())
    with open(".db-ignore-extensions.txt", "r") as f: unwanted_extensions = tuple(f.read().strip().splitlines())
    corpus_summary = []
    file_texts, metadatas = [], []
    for root, _, files in os.walk(temp_dir):
        for filename in files:
            if not is_unwanted_file(filename, unwanted_files, unwanted_extensions):
                file_path = os.path.join(root, filename)
                if Path(file_path).is_symlink(): continue
                if ".git" in file_path: continue
                with open(file_path, "r") as file:
                    if file_contains_secrets(file_path):
                        os.remove(file_path)
                        print(f"Deleted {file_path} as it contained secrets")
                        continue
                    print(f"Processing {file_path}")
                    try: file_contents = file.read()
                    except UnicodeDecodeError:
                        print(f"Skipping {file_path} as it could not be decoded")
                        continue
                    file_path = file_path.replace(temp_dir, "").lstrip("/")
                    corpus_summary.append({"file_name": file_path, "n_tokens": len(encoder.encode(file_contents, disallowed_special=()))})
                    file_texts.append(file_contents)
                    metadatas.append({"document_id": file_path})
    split_documents = create_documents_with_met(splitter, file_texts, metadatas = metadatas)


    print(f"Writing {len(split_documents)} documents to Pinecone")
    vector_store.from_documents(documents = split_documents, embedding = embeddings, index_name = os.environ["PINECONE_INDEX"], namespace = os.environ["NAMESPACE"], )
    Path("data").mkdir(parents = True, exist_ok = True)
    pd.DataFrame.from_records(corpus_summary).to_csv("data/corpus_summary.csv", index = False)

vector_store = Pinecone(index = pinecone.Index(os.environ["PINECONE_INDEX"]), embedding_function = embeddings.embed_query, text_key = "text", namespace = os.environ["NAMESPACE"])
splitter = RecursiveCharacterTextSplitter(chunk_size = int(os.environ["CHUNK_SIZE"]), chunk_overlap = int(os.environ["CHUNK_OVERLAP"]))
pinecone.init(api_key = os.environ["PINECONE_API_KEY"], environment = os.environ["ENVIRONMENT"])

def embed_into_db(repo_url, local_repo_path):
    pinecone_index = os.environ["PINECONE_INDEX"]
    namespace = os.environ["NAMESPACE"]
    index = pinecone.Index(pinecone_index)
    index.delete(delete_all = True, namespace = namespace)
    create_vector_db(repo_url, local_repo_path)

def embedding_search(query, k): return Pinecone.from_existing_index(os.environ["PINECONE_INDEX"], OpenAIEmbeddings(openai_api_key = os.environ["OPENAI_API_KEY"], openai_organization = os.environ["OPENAI_ORG_ID"], ), text_key = "text", namespace = os.environ["NAMESPACE"], ).similarity_search(query, k = k)

def create_vector_db(REPO_URL, LOCAL_REPO_PATH):
    clone_from_github(REPO_URL, LOCAL_REPO_PATH)
    process_file_list(LOCAL_REPO_PATH)
    return


if __name__ == "__main__":
    create_vector_db(os.environ["REPO_URL"], os.environ["LOCAL_REPO_PATH"])
