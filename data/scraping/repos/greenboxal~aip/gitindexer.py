import git
from git import Repo
from langchain.document_loaders import GitLoader
from langchain.docstore.document import Document


class GitIndexer:
    def __init__(self, repo_path):
        self.repo_path = repo_path

    def index_history(self, index, ref):
        repo = Repo(self.repo_path)
        docs = []

        for commit in repo.iter_commits(ref):
            metadata = {
                "commit_id": commit.hexsha,
                "commit_time": commit.committed_datetime.isoformat(),
                "commit_message": commit.message,
            }

            if commit.author:
                metadata["author_name"] = commit.author.name
                metadata["author_email"] = commit.author.email

            if len(commit.parents) > 0:
                parent = commit.parents[0]
            else:
                parent = git.NULL_TREE

            diff = commit.diff(other=parent, create_patch=True)

            doc = Document(page_content=diff, metadata=metadata)
            docs.append(doc)

        return docs

    def index_tree(self, index, ref):
        repo = Repo(self.repo_path)
        loader = GitLoader(self.repo_path)
        pass
