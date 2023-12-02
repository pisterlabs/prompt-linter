import requests
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from helpers.github import github_to_documents


class PortalLoader(BaseLoader):
    """Load Robocorp Portal example robot descriptions and code examples

    Args:
        url: Link to the public Robot index
    """

    def __init__(self, url: str):
        self.url = url

    def load(self) -> List[Document]:
        response = requests.get(self.url)

        if response.status_code == 200:
            json_data = response.json()
        else:
            raise Exception(f"Could not fetch Portal robot index file")

        output: List[Document] = []

        for robot in json_data:
            code_example = ""

            if robot["type"] != "unknown":
                taskFile = "tasks.py"
                codeType = "Python"

                if robot["type"] == "robot-framework":
                    taskFile = "tasks.robot"
                    codeType = "Robot Framework"

                task_response = requests.get(
                    f'https://github.com/{robot["repo"]}/blob/{robot["branch"]}/{taskFile}?raw=true'
                )

                if task_response.status_code == 200:
                    code_example = f"\n\nThis is the {codeType} code of that robot:\n```{task_response.text}```"

            document = Document(
                page_content=f'# Robot "{robot["name"]}"\n\n The robot is an example implementation for this task: {robot["description"]}.{code_example}',
                metadata={
                    "title": f'Example robot - {robot["name"]}',
                    "source": f'https://robocorp.com/portal/robot/{robot["repo"]}',
                },
            )
            output.append(document)

        return output