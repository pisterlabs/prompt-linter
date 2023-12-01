import tempfile
import subprocess
import os
from typing import List
import frontmatter

from langchain.docstore.document import Document
from helpers.markdown import markdown_to_documents


def github_to_documents(
    repo_url: str, file_ext: str, white_list: List[str]
) -> List[Document]:
    temp_dir = tempfile.mkdtemp()
    command = ["git", "clone", "--depth=1", repo_url, temp_dir]
    subprocess.run(command)

    output = []

    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(file_ext):
                for pattern in white_list:
                    subpath = os.path.join(*pattern.split("/"))
                    filepath = os.path.join(root, file)
                    if subpath in filepath:
                        post = frontmatter.load(filepath)
                        documents = markdown_to_documents(post.content)

                        for doc in documents:
                            doc.metadata["source"] = filepath
                            if "title" in post:
                                doc.metadata[
                                    "title"
                                ] = f"{post['title']} - {post['description']}"

                        output.extend(documents)

    return output, temp_dir