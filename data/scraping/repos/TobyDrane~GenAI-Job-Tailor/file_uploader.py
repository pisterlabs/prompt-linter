import os
import pathlib
import json

from typing import List

import pandas as pd

from langchain.document_loaders import UnstructuredFileLoader
from streamlit.runtime.uploaded_file_manager import UploadedFile

from core.models.file import File


class FileUploader:
    def __init__(self, data_folder: str) -> None:
        self.data_folder = data_folder

    def upload_file(self, uploaded_file: UploadedFile) -> File:
        file_name = uploaded_file.name
        upload_path = os.path.join(self.data_folder, "upload", file_name)
        file_type = pathlib.Path(upload_path).suffix

        file = File(
            path=upload_path,
            type=file_type,
            name=file_name,
        )

        with open(file.path, "wb") as f:
            f.write(uploaded_file.getvalue())

        file.upload_file_json(self.data_folder)
        return file

    def get_uploaded_files(self) -> List[str]:
        upload_folder = os.path.join(self.data_folder, "upload")
        files = os.listdir(upload_folder)
        return files

    def get_uploaded_file_contents(self, file_name: str) -> str:
        upload_path = os.path.join(self.data_folder, "file", f"{file_name}.json")
        with open(upload_path, "r") as f:
            data = json.load(f)

        document = UnstructuredFileLoader(data["path"]).load()
        return document[0].page_content
