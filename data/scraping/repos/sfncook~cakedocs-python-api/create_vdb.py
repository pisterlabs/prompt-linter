import os
from clone_repo import clone_repo
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from utils import get_repo_name_from_url

def get_file_chunks(filename):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )
    js_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS,
        chunk_size=1500,
        chunk_overlap=200,
    )
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(filename)
    elif filename.endswith(".md"):
        loader = UnstructuredMarkdownLoader(filename)
    elif filename.endswith(".html"):
        loader = UnstructuredHTMLLoader(filename)
    else:
        loader = TextLoader(filename)
    document = loader.load()
    if filename.endswith(".js"):
        chunks = js_splitter.split_documents(document)
    else:
        chunks = text_splitter.split_documents(document)
    print(f"Split {filename} into {len(chunks)} chunks")
    return chunks

def get_dir_chunks_recursively(repo_clone_dir):
    count = 0
    ignore_list = ['.git', 'node_modules', '__pycache__', '.idea', '.vscode', 'package-lock.json', 'yarn.lock']
    chunks = []
    for root, dirs, files in os.walk(repo_clone_dir):
        # Only process first directory - For testing/debugging
#         if count > 1:
#             continue
#         count += 1

        dirs[:] = [d for d in dirs if d not in ignore_list]  # modify dirs in-place
        for file in files:
            if file in ignore_list:
                continue
#             print(f"Processing file {file}")
            filepath = os.path.join(root, file)
            try:
                chunks.extend(get_file_chunks(filepath))
            except Exception as e:
                print(f"Failed to process {filepath} due to error: {str(e)}")
    return chunks

def store_chunks_in_pinecone(chunks, pinecone_index_name, repo_name):
    print("Storing chunks in Pinecone...")
    embeddings = OpenAIEmbeddings(disallowed_special=())
    pinecone_index = pinecone.Index(pinecone_index_name)
    pinecone_vdb = Pinecone(pinecone_index, embeddings.embed_query, "cakedocs", namespace=repo_name)
    pinecone_vdb.delete(delete_all=True, namespace=repo_name)
    print("Done deleting previous chunks from Pinecone!")
    print(f"Adding chunks for {repo_name} to Pinecone...")
    pinecone_vdb.add_documents(chunks)
    print("Done storing chunks in Pinecone!")

def create_vdb(repo_url, code_repo_path, temp_dir, pinecone_index_name):
    repo_name = get_repo_name_from_url(repo_url)

    repo_clone_dir = clone_repo(repo_url, code_repo_path)
    chunks = get_dir_chunks_recursively(repo_clone_dir)
    store_chunks_in_pinecone(chunks, pinecone_index_name, repo_name)

    print("VDB generated!")
    return True
