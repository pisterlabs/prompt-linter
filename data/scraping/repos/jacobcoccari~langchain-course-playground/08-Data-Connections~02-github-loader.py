from git import Repo
from langchain.document_loaders import GitLoader

# repo = Repo.clone_from(
#     "https://github.com/langchain-ai/langchain",
#     to_path="./08-Data-Connections/langchain",
# )
# branch = repo.head.reference

loader = GitLoader(repo_path="./08-Data-Connections/langchain", branch="master")
data = loader.load()
print(data[3])
print(len(data))
