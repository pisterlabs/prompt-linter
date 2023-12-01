# Set env var OPENAI_API_KEY or load from a .env file
import dotenv
# import langchain

import os

repo_path = os.getcwd()
print("Current repo_path:", repo_path)

import sys
print(sys.path)
print(sys.executable)


dotenv.load_dotenv()

# from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser



# Load
loader = GenericLoader.from_filesystem(
    repo_path+"/libs/langchain/langchain",
    glob="**/*",
    suffixes=[".php", ".js", ".css"],
    parser=LanguageParser(language=Language.PHP, parser_threshold=500)
)
documents = loader.load()
len(documents)
