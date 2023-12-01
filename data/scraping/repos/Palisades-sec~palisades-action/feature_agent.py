import base64
import json
import os
from uuid import uuid4

import requests
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.vectorstores import FAISS

from schemas import GitHubIssue

llm = ChatOpenAI(
    max_tokens=1000,
    model="gpt-4",
    temperature=0,
    openai_api_key=os.environ["OPENAI_API_KEY"],
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# TODO
repository_name = os.environ["REPO_NAME"]
cf_auth_token = os.environ["CF_AUTH_TOKEN"]
token = os.environ["GITHUB_TOKEN"]
headers = {"Authorization": f"token {token}"}


def get_issues(repository_name, issue_number):
    url = f"https://api.github.com/repos/{repository_name}/issues/{issue_number}"
    res = requests.get(url, headers=headers)
    # TODO
    # if res.status_code == 200:
    # issues = json.loads(res.content)
    # issues = json.loads(res.content)
    # issue_class_list = []
    # for issue in issues:
    #     issue_class_list.append(GitHubIssue(**issue))
    # return issue_class_list
    issue = json.loads(res.content)
    issue = GitHubIssue(**issue)
    issue = f"Title: {issue.title}\n\n{issue.body}"
    return issue


# TODO Docstring
@tool
def generate_code(issue_number: str):
    """Generates code from the given description and returns it along with target file path to resolve the issue. Accepts the issue number as it's input."""
    issue = get_issues(repository_name, issue_number)
    FAISS_DB_PATH = f"./FAISS/{os.path.basename(repository_name)}"
    faiss_db = FAISS.load_local(FAISS_DB_PATH, embeddings)

    retrieved_docs = faiss_db.similarity_search(issue, k=4)
    retrieved_content = [
        f"From path {docs.metadata['source']}\n{docs.page_content}"
        for docs in retrieved_docs
    ]
    retrieved_content = "\n".join(retrieved_content)
    body = {"issue": issue, "retrieved": retrieved_content}
    # TODO
    url = "https://us-central1-palisades-sec.cloudfunctions.net/palisade-feature"
    res = requests.post(
        url,
        data=json.dumps(body),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": cf_auth_token,
        },
    )
    if res.status_code == 401:
        raise Exception("Authorization for cloud function failed")
    response = json.loads(res.content)
    file_content = response["file_content"]
    file_path = response["file_path"]
    pr_data = response["pr_data"]
    return file_content, file_path, pr_data


@tool
def publish_changes(file_content: str, file_path: str):
    """TODO Docstring, Creates new branch and publishes changes in the new branch. Accepts a filepath and content of the file. Returns name of the new branch"""
    #  Get main branch sha
    print("Getting main branch sha")
    url = f"https://api.github.com/repos/{repository_name}/git/ref/heads/main"
    res = requests.get(url, headers=headers)
    sha = json.loads(res.content)["object"]["sha"]

    #  Create new branch
    print("Creating new branch")
    new_branch_name = f"test_branch_{uuid4()}"
    url = f"https://api.github.com/repos/{repository_name}/git/refs"
    body = {"ref": f"refs/heads/{new_branch_name}", "sha": sha}
    res = requests.post(url, data=json.dumps(body), headers=headers)

    #  Get file SHA
    print("Getting file SHA")
    url = f"https://api.github.com/repos/{repository_name}/contents/{file_path}"
    res = requests.get(url, headers=headers)
    sha = json.loads(res.content)["sha"]

    # Update file
    print("Updating file")
    content = base64.b64encode(file_content.encode("ascii"))
    body = {
        "message": "commit from palisade",
        "content": content.decode("ascii"),
        "branch": new_branch_name,
        "sha": sha,
    }
    res = requests.put(url, data=json.dumps(body), headers=headers)
    return new_branch_name


@tool
def create_pr(head: str, pr_title: str, pr_body: str):
    """TODO Docstring, Creates a pull request between given branch name and main branch. Requires branch name, PR title and PR body."""
    url = f"https://api.github.com/repos/{repository_name}/pulls"
    request_body = {
        "title": pr_title,
        "body": pr_body,
        "head": head,
        "base": "main",
    }
    res = requests.post(url, data=json.dumps(request_body), headers=headers)
    match res.status_code:
        case 201:
            return "PR created successfull."
        case _:
            return json.loads(res.content)["message"]


tools = [generate_code, publish_changes, create_pr]

agent_prompt = """You are an assistant. Whenever a new issue is created in a GitHub repository your task is to generate code that resolves the given issue, commit these changes to a new branch and create a pull request."""


feature_development_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
