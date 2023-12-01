from github import Github
from langchain.tools import tool


def sweep_issue_tool_factory(
    repository_full_name: str,
    github_access_token: str,
):
    @tool
    def create_sweep_issue(
        title: str,
        body: str,
    ):
        github = Github(github_access_token)
        github_repo = github.get_repo(repository_full_name)
        github_issue = github_repo.create_issue(
            title=f"Sweep: {title}",
            body=body,
        )

    return create_sweep_issue
