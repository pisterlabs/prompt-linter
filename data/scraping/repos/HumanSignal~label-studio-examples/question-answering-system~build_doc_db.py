import os
import argparse
from langchain.document_loaders.git import GitLoader
from git import Repo
from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma



def main(repo_path, repo_url, persist_directory):

    if not os.path.exists(repo_path):
        print("Cloning %s into %s" % (repo_url, repo_path))
        repo = Repo.clone_from(
                repo_url,
                to_path=repo_path)
    else:
        repo = Repo(repo_path)
    
    branch = repo.head.reference

    print("Loading repository document data.")
    loader = GitLoader(
            repo_path=repo_path,
            branch=branch,
            file_filter=lambda f: f.endswith('.md'))
    data = loader.load()

    print("Splitting repository documents into chunks of 500")
    text_splitter = MarkdownTextSplitter(
            chunk_size=500,
            chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    print("Computing and loading OpenAI embeddings into Chroma vectorstore.")
    vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory)
    vectorstore.persist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameterized script for processing markdown files from a Github repo and storing OpenAI embeddings in a chromadb.")
    
    parser.add_argument("--repo_path", default="./example_data/label-studio-repo", help="Path to the repository.")
    parser.add_argument("--repo_url", default="https://github.com/HumanSignal/label-studio", help="URL of the repository to clone.")
    parser.add_argument("--persist_directory", default="pd", help="Directory to persist the embeddings.")

    args = parser.parse_args()
    
    main(args.repo_path, args.repo_url, args.persist_directory)
