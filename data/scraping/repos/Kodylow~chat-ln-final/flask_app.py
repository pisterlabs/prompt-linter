
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, request, jsonify
import requests
import shutil
import os
import tempfile
import subprocess
from langchain.document_loaders import UnstructuredMarkdownLoader, PyPDFLoader, ReadTheDocsLoader, GitLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import glob

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = 'us-east4-gcp'

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
index_name = "chatln"

def is_zero_content(content, threshold=0.9):
    zero_count = content.count('0')
    if zero_count / len(content) >= threshold:
        print("Zero content detected")
        return True
    return False


def load_markdown_files(repo_path):
    markdown_files = glob.glob(os.path.join(repo_path, "**/*.md"), recursive=True)
    return markdown_files

@app.route('/load_documents', methods=['POST'])
def load_documents():
    data = request.get_json()
    url = data.get('url')
    loader_type = data.get('loader_type')
    branch = data.get('branch')

    with tempfile.TemporaryDirectory() as temp_dir:
        # subprocess.run(["git", "clone", url, temp_dir])

        if loader_type == 'github':
            print("Running git loader")
            loader = GitLoader(
                clone_url=url,
                repo_path=temp_dir,
                branch=branch,
                file_filter=lambda file_path: file_path.endswith(".asciidoc")
            )
            print("Loading documents")
            data = loader.load()
        elif loader_type == 'pdf':
            loader = PyPDFLoader(url)
            data = loader.load()
        elif loader_type == 'readTheDocs':
            loader = ReadTheDocsLoader(url)
            data = loader.load()
        else:
            return jsonify({'error': 'Invalid loader type'}), 400

        print("Running text splitter")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=1000)
        texts = text_splitter.split_documents(data)
        # Filter out texts with too many zeros
        texts = [t for t in texts if not is_zero_content(t.page_content)]

        # Update metadata for each text
        for text in texts:
            text.metadata['url'] = url

        print("Texts:", texts[0])
        metadatas = [t.metadata for t in texts]
        docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name, metadatas=metadatas)

    return jsonify({'message': 'Documents loaded successfully'}), 200


if __name__ == '__main__':
    app.run(debug=True)
