import os
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from helpers.github import github_to_documents


class RoboLoader(BaseLoader):
    """Load Robo Python automation framework documentation

    Args:
        repo_url: Link to the Github repository
        white_list: Path paterns to be included in docs
    """

    def __init__(self, repo_url: str, white_list: List[str]):
        self.repo_url = repo_url
        self.white_list = white_list

    def load(self) -> List[Document]:
        documents, temp_dir = github_to_documents(self.repo_url, "md", self.white_list)

        for doc in documents:
            doc.metadata["source"] = self._get_source(doc.metadata["source"], temp_dir)
            doc.metadata["title"] = "Robo Python automation framework"

        return documents

    def _get_source(self, file_name: str, dir: str):
        paths = file_name.replace(dir, "").replace("README.md", "").split(os.path.sep)
        url_path = "/".join(paths)

        url = f"https://github.com/robocorp/robo/blob/master{url_path}"

        return url