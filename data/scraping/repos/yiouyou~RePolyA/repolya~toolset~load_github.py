from langchain.document_loaders import GitLoader
from langchain.document_loaders import GitHubIssuesLoader
import os


##### GitLoader
def load_git_to_docs(clone_url: str, repo_path: str, branch: str, file_extensions: tuple[str, ...] = (".py",)):
    loader = GitLoader(
        clone_url=clone_url,
        repo_path=repo_path,
        branch=branch,
        file_filter=lambda file_path: file_path.endswith(file_extensions),
    )
    docs = loader.load()
    return docs


##### GitHubIssuesLoader
def load_github_issues_to_docs(repo: str):
    loader = GitHubIssuesLoader(
        repo=repo,
        access_token=os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
    )
    docs = loader.load()
    return docs

