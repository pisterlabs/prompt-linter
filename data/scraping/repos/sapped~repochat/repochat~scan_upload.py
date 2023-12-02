# found here: https://www.reddit.com/r/ChatGPTPro/comments/14v3y03/chatgpt_code_interpreter_for_github_repo_analysis/
# repurposed this guy: https://python.langchain.com/docs/use_cases/code/twitter-the-algorithm-analysis-deeplake

import os
import pathspec

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from repochat.config import settings


async def run_index():
    embeddings = OpenAIEmbeddings(disallowed_special=())

    # index the codebase
    root_dir = "./"
    docs = []

    # Dictionary to store the .gitignore rules for each subdirectory
    ignore_specs = {}

    # Loop through all first-level subdirectories
    for subdir in next(os.walk(root_dir))[1]:
        gitignore_file = os.path.join(root_dir, subdir, ".gitignore")
        if os.path.exists(gitignore_file):
            with open(gitignore_file, "r") as file:
                # Compile the .gitignore rules into a pathspec and store it in the dictionary
                ignore_patterns = file.read().splitlines()
                ignore_specs[subdir] = pathspec.PathSpec.from_lines(
                    pathspec.patterns.GitWildMatchPattern, ignore_patterns
                )

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip files in the .git directory
        if ".git" in dirpath:
            continue
        for filename in filenames:
            subdir = dirpath.split(os.sep)[
                1
            ]  # Get the first-level subdirectory of the file
            subdir_path = os.path.join(root_dir, subdir)
            filepath = os.path.join(dirpath, filename)[
                len(subdir_path) + 1 :
            ]  # Make the file path relative to the subdirectory
            if subdir in ignore_specs and ignore_specs[subdir].match_file(filepath):
                # This file matches a .gitignore pattern from its subdirectory, so skip it
                continue
            try:
                loader = TextLoader(os.path.join(dirpath, filename))
                docs.extend(loader.load_and_split())

            except Exception as e:
                pass

    # chunk the files
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    # execute the indexing
    ds = DeepLake(
        dataset_path=f"hub://{settings.deeplake_username}/{settings.deeplake_dataset_name}",
        embedding_function=embeddings,
        overwrite=True,
    )
    ds.delete(delete_all=True)

    ds.add_documents(texts)
    return 0
