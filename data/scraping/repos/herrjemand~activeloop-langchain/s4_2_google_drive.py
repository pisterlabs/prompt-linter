from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.document_loaders import GoogleDriveLoader

loader = GoogleDriveLoader(
    folder_id="1_GFLJmdWkk1baXsB8I4wZc6VeidEBSbz",
    recursive=False,
)

docs = loader.load()