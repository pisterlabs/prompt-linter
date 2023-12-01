# import the required libraries and modules. It is important to install them using pip beforehand.

import argparse
import base64
import fnmatch
import os

import requests
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import config

# Setup constants and API keys

GITHUB_TOKEN = config.GITHUB_ACCESS_TOKEN

# Definition of utility functions


## Parses the given GitHub URL to get the owner and name from the url


def parse_github_url(url):
    parts = url.strip("/").split("/")
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo


## Fetches all files from the specified GitHub repository and folder using the GitHub API.


def get_files_from_github_repo(owner, repo, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content["tree"]
    else:
        raise ValueError(f"Error fetching repo contents: {response.status_code}")


## Retrieves the contents of Markdown files from the list of files.


def fetch_md_contents(files):
    md_contents = []
    for file in files:
        if file["type"] == "blob" and fnmatch.fnmatch(file["path"], "*.md"):
            response = requests.get(file["url"])
            if response.status_code == 200:
                content = response.json()["content"]
                decoded_content = base64.b64decode(content).decode("utf-8")
                print("Fetching Content from ", file["path"])
                md_contents.append(
                    Document(
                        page_content=decoded_content, metadata={"source": file["path"]}
                    )
                )
            else:
                print(f"Error downloading file {file['path']}: {response.status_code}")
    return md_contents


## Processes the contents of the Markdown files into smaller chunks for further processing.


def get_source_chunks(files):
    print("In get_source_chunks ...")
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in fetch_md_contents(files):
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadate=source.metadata))
    return source_chunks


# Create the main() function that will be responsible for executing the main logic of the script.


def main():
    parser = argparse.ArgumentParser(
        description="Fetch all *.md files from a GitHub repository."
    )
    parser.add_argument(
        "https://github.com/dfcantor/obsidian-vault-sync/tree/main/Obsidian%20Vault",
        help="GitHub repository URL",
    )

    args = parser.parse_args()

    GITHUB_OWNER, GITHUB_REPO = parse_github_url(args.url)

    all_files = get_files_from_github_repo(GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN)

    CHROMA_DB_PATH = f"./chroma/{os.path.basename(GITHUB_REPO)}"

    chroma_db = None

    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Creating Chroma DB at {CHROMA_DB_PATH}...")
        source_chunks = get_source_chunks(all_files)
        chroma_db = Chroma.from_documents(
            source_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH
        )
        chroma_db.persist()
    else:
        print(f"Loading Chroma DB from {CHROMA_DB_PATH} ... ")
        chroma_db = Chroma(
            persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings()
        )

    qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
    qa = RetrievalQA(
        combine_documents_chain=qa_chain, retriever=chroma_db.as_retriever()
    )

    while True:
        print("\n\n\033[31m" + "Ask a question" + "\033[m")
        user_input = input()
        print("\033[31m" + qa.run(user_input) + "\033[m")


# Executing the main function

if __name__ == "__main__":
    main()
