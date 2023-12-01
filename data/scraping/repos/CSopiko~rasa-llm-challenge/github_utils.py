import base64
import time

import requests
from langchain.docstore.document import Document


def get_files_from_github_repo(owner, repo, token, tag_sha="main"):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{tag_sha}?recursive=1"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content.get("tree")
    else:
        raise ValueError(
            f"Error fetching {repo} contents: check {tag_sha} tag exists. response: {response.status_code}"
        )


def fetch_mdx_contents(
    mdx_files, wait_for_renewal=False, max_requests=60, verbose=False
):
    mdx_contents = []
    if not wait_for_renewal:
        return mdx_contents
    for i, file in enumerate(mdx_files):

        if i == max_requests and wait_for_renewal:
            print(f"Waiting for token renewal 3600 seconds...")
            time.sleep(1800)
            print(f"Halfway there...")
            time.sleep(1800)
            print(f"Resuming...")

        response = requests.get(file["url"])
        if response.status_code == 200:
            response = response.json()
            content = response.get("content")
            decoded_content = base64.b64decode(content).decode("utf-8")
            if verbose:
                print("Fetching Content from ", file["path"])
            mdx_contents.append(
                Document(
                    page_content=decoded_content, metadata={"source": file.get("path")}
                )
            )
        elif verbose:
            print(f"Error downloading file {file['path']}: {response.status_code}")
    return mdx_contents


def load_mdx_contents(mdx_path):
    print(f"Using old mdx files from disk.")
    import pickle

    with open(mdx_path, "rb") as f:
        mdx_contents = pickle.load(f)
    return mdx_contents


def save_mdx_content(mdx_path, mdx_contents):
    print(f"Saving mdx files to disk.")
    import pickle

    with open(mdx_path, "wb") as f:
        pickle.dump(mdx_contents, f)
    return mdx_contents
