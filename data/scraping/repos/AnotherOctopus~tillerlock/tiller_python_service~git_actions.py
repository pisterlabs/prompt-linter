from git import Repo, GitCommandError
import git
from static_vals import GITHUB_TOKEN 
import subprocess
import tempfile
import uuid
import requests
import json
import os
import re
import openai

def clone_repo(url):
    """
    Clone a git repository from a specified URL into a randomly named directory within /tmp.

    Parameters:
        url (str): The URL of the git repository.

    Returns:
        str: The full path to the cloned repository.
    """
    # Create a temporary directory.
    target_dir = tempfile.mkdtemp(dir="/tmp")

    # Clone the repository.
    repo_name = url.split("/")[-1].replace(".git", "")
    full_repo_path = os.path.join(target_dir, repo_name)
    clone_url = url.replace("https://", f"https://x-access-token:{GITHUB_TOKEN}@")
    result = subprocess.run(["git", "clone", clone_url, full_repo_path])

    if result.returncode != 0:
        print(f"Error cloning the repository. Return code: {result.returncode}")
        return None
    return full_repo_path

def git_add_all(repo_path):
    try:
        repo = Repo(repo_path)
        repo.git.add(A=True)  # Adds all changes
        return "All files added successfully"
    except GitCommandError as e:
        return str(e)

def git_commit(repo_path, message):
    try:
        repo = Repo(repo_path)
        repo.index.commit(message)
        return "Commit successful with message: {}".format(message)
    except GitCommandError as e:
        return str(e)

def git_push(repo_path, branch='master'):
    try:
        repo = Repo(repo_path)
        origin = repo.remote(name='origin')
        origin.push(branch)
        return "Push to {} successful".format(branch)
    except GitCommandError as e:
        return str(e)

def git_fetch_all(repo_path):
    try:
        repo = Repo(repo_path)
        for remote in repo.remotes:
            remote.fetch()
        return "Fetch all successful"
    except GitCommandError as e:
        return str(e)

def git_checkout_branch(repo_path, branch_name):
    try:
        repo = Repo(repo_path)
        repo.git.checkout(branch_name)
        return "Switched to branch: {}".format(branch_name)
    except GitCommandError as e:
        return str(e)

def generate_random_branch_name():
    """
    Generate a random branch name.

    Returns:
        str: The generated branch name.
    """
    return "branch-" + str(uuid.uuid4())

def clone_and_create_new_branch(repo_url, initial_branch):
    """
    Clone a repository, switch to a specified branch, and create a new branch with a random name.

    Examples:
        branch_name, repo_path = clone_and_create_new_branch("git@github.com:AnotherOctopus/tillerlock.git", "git-functions")

    Parameters:
        repo_url (str): The URL of the git repository.
        initial_branch (str): The name of the branch to switch to after cloning.

    Returns:
        tuple: A tuple containing the branch name and repo path. If an error occurs, returns an error message.
    """
    # Clone the repo
    repo_path = clone_repo(repo_url)
    if repo_path is None:
        return "Failed to clone repository"

    # Switch to the initial branch
    checkout_message = git_checkout_branch(repo_path, initial_branch)
    if not checkout_message.startswith("Switched"):
        return "Failed to switch to initial branch: " + checkout_message

    # Create a new branch with a random name and switch to it
    new_branch_name = generate_random_branch_name()
    try:
        repo = Repo(repo_path)
        repo.git.checkout('-b', new_branch_name)
        return new_branch_name, repo_path
    except GitCommandError as e:
        print(e)
        return "Failed to switch to new branch: " + str(e)

def git_add_commit_push(repo_path, commit_message):
    """
    Add all changes, commit them, and push to the remote repository.

    Parameters:
        repo_path (str): The full path to the local git repository.
        commit_message (str): The commit message.

    Returns:
        str: A message about the operation's success or the error message.
    """
    # Add all changes
    add_message = git_add_all(repo_path)
    if not add_message.startswith("All files"):
        return "Failed to add changes: " + add_message

    # Commit changes
    commit_message = git_commit(repo_path, commit_message)
    if not commit_message.startswith("Commit"):
        return "Failed to commit changes: " + commit_message

    # Push changes
    current_branch = Repo(repo_path).active_branch.name
    push_message = git_push(repo_path, current_branch)
    if not push_message.startswith("Push"):
        return "Failed to push changes: " + push_message

    return "Add, commit, and push operations were successful"

def open_pull_request(repo_url, source_branch, target_branch):
    print("Opening pull request from {} to {}".format(source_branch, target_branch))
    owner, repo = parse_repo_url(repo_url)

    # Create the URL for the pull request
    pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"

    # Get the GitHub token from the environment
    github_token = GITHUB_TOKEN
    if github_token is None:
        print("Please set your GitHub token in the GITHUB_TOKEN environment variable.")
        return

    # Define the headers
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Define the data for the pull request
    data = {
        "title": f"Pull request from {source_branch} to {target_branch}",
        "head": source_branch,
        "base": target_branch
    }

    # Send the request to create the pull request
    response = requests.post(pr_url, headers=headers, data=json.dumps(data))
    pr_number = response.json()['number']

    # If the request was successful, print the URL of the new pull request
    if response.status_code == 201:
        body = "This is an AI created pull request"
        diff_url = response.json()['diff_url']
        diff_response = requests.get(diff_url, headers=headers)
        print(diff_response.content)
        if diff_response.status_code == 200:
            print(str(diff_response.content))
        else:
            print(f"Failed to create body")
        body_update_data = {
            "body": body
        }
        # Send the request to create the pull request
        print(f"Updating body of pull request {pr_number}")
        requests.post(pr_url + f"/{pr_number}", headers=headers, data=json.dumps(body_update_data))

        return response.json()['html_url']
    else:
        print(f"Failed to create pull request: {response.content}")

def merge_pull_request(pull_request_url, commit_title, commit_message, merge_method='merge'):
    owner, repo, pull_number = parse_pull_request_url(pull_request_url)
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/merge"
    print("merging: ", url)
    github_token = os.getenv("GITHUB_TOKEN")
    headers = {
        'Authorization': f"token {github_token}",
        'Accept': 'application/vnd.github.v3+json',
    }
    data = {
        'commit_title': commit_title,
        'commit_message': commit_message,
        'merge_method': merge_method,  # can be 'merge', 'squash', or 'rebase'
    }
    response = requests.put(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        print('Pull request merged successfully.')
    else:
        print(f'Failed to merge pull request. Response: {response.content}')

def parse_repo_url(repo_url):
    # Pattern to match the username and repository name
    pattern = r'github\.com:(\w+)/(\w+)\.git'

    # Search for the pattern in the repo_url
    match = re.search(pattern, repo_url)

    if match:
        username = match.group(1)
        repository = match.group(2)
        return username, repository
    else:
        return None, None

def parse_pull_request_url(pull_request_url):
    # Pattern to match the username and repository name
    pattern = r'github\.com\/(\w+)/(\w+)\/pull\/(\w+)'

    # Search for the pattern in the repo_url
    match = re.search(pattern, pull_request_url)

    if match:
        username = match.group(1)
        repository = match.group(2)
        pull_number = match.group(3)
        return username, repository, pull_number
    else:
        return None, None, None

# # Example usage
# repo_url = "git@github.com:AnotherOctopus/tillerlock.git"
# username, repository = parse_repo_url(repo_url)
# print("Username:", username)
# print("Repository:", repository)

# repo_url="git@github.com:AnotherOctopus/tillerlock.git"
# source_branch="branch-fcdf932e-9134-489c-95bd-80e3061d598b"
# target_branch="some-change"
#
# usage
# link = open_pull_request(
#     repo_url=repo_url,
#     source_branch=source_branch,
#     target_branch="main"
# )
#
# print(link)

# link = 'https://github.com/AnotherOctopus/tillerlock/pull/10'

# branch, url = clone_and_create_new_branch("git@github.com:AnotherOctopus/tillerlock.git", "git-functions")
# print(branch, url)

# merge_pull_request(link, "title", "message")