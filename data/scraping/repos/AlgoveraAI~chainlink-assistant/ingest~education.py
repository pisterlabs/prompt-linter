import pickle
import fnmatch
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from langchain.docstore.document import Document

from config import DATA_DIR, get_logger

logger = get_logger(__name__)


def load_gh_data(gh_tuple: tuple, chunk: bool = True, chunk_size: int = 1200):
    repo_url = gh_tuple[0]
    wildcards = gh_tuple[1]
    documents = get_github_docs(repo_url, wildcards)

    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 1200,
    #     chunk_overlap  = 20,
    # )
    # documents = splitter.split_documents(documents)

    return documents


def get_github_docs(repo_url: str, wildcards: Optional[List[str]] = None):
    repo_url = repo_url.replace(".git", "")
    url_parts = repo_url.split("/")
    if len(url_parts) < 5 or not url_parts[2].endswith("github.com"):
        raise ValueError("Invalid GitHub URL format")

    repo_owner = url_parts[3]
    repo_name = url_parts[4]

    if len(url_parts) > 6 and url_parts[5] == "tree":
        branch = "/".join(url_parts[6:])
    else:
        branch = None

    repo_url = f"https://github.com/{repo_owner}/{repo_name}"
    if not repo_url.endswith(".git"):
        repo_url += ".git"

    with tempfile.TemporaryDirectory() as d:
        if branch is not None:
            git_command = f"git clone --depth 1 -b {branch} {repo_url} ."
        else:
            git_command = f"git clone --depth 1 {repo_url} ."

        subprocess.check_call(
            git_command,
            cwd=d,
            shell=True,
        )
        git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d)
            .decode("utf-8")
            .strip()
        )

        repo_path = Path(d)
        markdown_files = list(repo_path.glob("**/*.md")) + list(
            repo_path.glob("**/*.mdx")
        )
        if wildcards is not None:
            wildcards = [
                f"*{pattern}/*"
                if not pattern.startswith("*") and not pattern.endswith("/*")
                else pattern
                for pattern in wildcards
            ]
            filtered_files = []
            for wildcard in wildcards:
                filtered_files.extend(
                    file
                    for file in markdown_files
                    if fnmatch.fnmatch(str(file), wildcard)
                )
            markdown_files = list(set(filtered_files))  # Remove duplicates

        markdown_files = [
            file for file in markdown_files if file.name.endswith("course.md")
        ]

        documents = []
        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                read = f.read()
                documents.append(
                    Document(
                        page_content=read,
                        metadata={"source": github_url, "type": "education"},
                    )
                )

    return documents


def scrap_education_docs():
    gh_tuple = (
        "https://github.com/oceanByte/chainlink-education.git",
        "src/api/src/shared/course",
    )
    chainlink_education_documents = load_gh_data(gh_tuple)

    # Save the documents to a pickle file with date in the name
    with open(f"{DATA_DIR}/education_documents.pkl", "wb") as f:
        pickle.dump(chainlink_education_documents, f)

    logger.info(f"Scrapped chainlink education documents.")

    return chainlink_education_documents
