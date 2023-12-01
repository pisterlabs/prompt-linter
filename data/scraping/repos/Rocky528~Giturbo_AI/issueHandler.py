from github import Github
from github import Auth
import config
from github_lib import get_github_file_content, get_github_links_from_issue_content
from openai_lib import get_completion
import requests
import openai

def get_github_repos(payload):
    auth = payload['github_token']
    username = payload['username']
    g = Github(auth)
    user = g.get_user(username)
    repositories = user.get_repos()
    
    repo_names = []
    for repository in repositories:
        repo_names.append(repository.name)
    
    return {"repositories": repo_names}

def create_issue(payload):
    auth = payload['github_token']
    g = Github(auth)
    username = payload['username']
    repo_name = payload['repository']['name']
    repo = g.get_repo(username + '/' + repo_name)
    title = payload['repository']['issue']['title']
    body = payload['repository']['issue']['body']
    issue = repo.create_issue(title=title, body=body)

    response = {
        'status': 'success',
        'title': issue.title,
        'body': issue.body,
        'url': issue.html_url
    }
    
    return response

def get_possible_labels(payload):
    auth = payload['github_token']
    g = Github(auth)
    username = payload['username']
    repo_name = payload['repository']['name']

    # Fetch the repository object
    repo = g.get_repo(username + '/' + repo_name)

    # Get the labels for the repository
    label_data = repo.get_labels()

    # Extract the label names
    possible_labels = [label.name for label in label_data]

    return possible_labels

def get_all_issues(payload):
    auth = payload['github_token']
    g = Github(auth)
    username = payload['username']
    repo_name = payload['repository']['name']

    # Fetch the repository object
    repo = g.get_repo(username + '/' + repo_name)

    # Get all issues of the repository
    issues = repo.get_issues(state='all')

    issue_data = []
    
    # Iterate over the issues and extract their title and number
    for issue in issues:
        issue_dict = {
            'title': issue.title,
            'number': issue.number
        }
        issue_data.append(issue_dict)
    
    return issue_data
#///
def get_existing_labels(payload):
    auth = payload['github_token']
    g = Github(auth)
    username = payload['username']
    repo_name = payload['repository']['name']
    
    # Fetch the repository and issue objects
    repo = g.get_repo(username + '/' + repo_name)
    issue_number = payload['repository']['issue']['number']
    issue = repo.get_issue(issue_number)

    # Get the list of labels for the issue
    labels = issue.get_labels()

    existing_labels = []
    
    # Iterate over the labels and extract their names
    for label in labels:
        existing_labels.append(label.name)
    
    return existing_labels
#///

def add_label_to_issue(payload):
    auth = payload['github_token']
    g = Github(auth)
    username = payload['username']
    repo_name = payload['repository']['name']
    issue_number = payload['repository']['issue']['number']
    labels = payload['repository']['issue']['labels']

    # Fetch the repository object
    repo = g.get_repo(username + '/' + repo_name)

    # Get the issue object
    issue = repo.get_issue(issue_number)

    # Add the label to the issue
    issue.add_to_labels(labels)

    return "Label added successfully."

def leave_comment_on_issue(payload):
    auth = payload['github_token']
    g = Github(auth)
    username = payload['username']
    repo_name = payload['repository']['name']
    issue_number = payload['repository']['issue']['number']
    comment_body = payload['repository']['issue']['comment_body']

    # Fetch the repository object
    repo = g.get_repo(username + '/' + repo_name)

    # Get the issue object
    issue = repo.get_issue(issue_number)

    # Leave a comment on the issue
    issue.create_comment(comment_body)

    return "Comment added successfully."

def get_all_pull_requests(payload):
    auth = payload['github_token']
    g = Github(auth)
    username = payload['username']
    repo_name = payload['repository']['name']

    # Fetch the repository object
    repo = g.get_repo(username + '/' + repo_name)

    # Get all the pull requests in the repository
    pull_requests = repo.get_pulls(state='all')

    pulls = []
    for pull_request in pull_requests:
        pull_dict = {
            'title': pull_request.title,
            'number': pull_request.number
        }
        pulls.append(pull_dict)
    return pulls

def merge_pull_request(payload):
    auth = payload['github_token']
    g = Github(auth)
    username = payload['username']
    repo_name = payload['repository']['name']
    pr_number = payload['repository']['pull_request']['number']

    # Fetch the repository object
    repo = g.get_repo(username + '/' + repo_name)

    # Get the pull request object
    pull_request = repo.get_pull(pr_number)

    # Merge the pull request
    pull_request.merge()
    return "Pull request merged successfully."

def handle_issue(payload):
    # TODO: Implement logic for handling issues
    issue_content = payload["issue_content"]
    owner = payload["owner"]
    repo = payload["repo"]
    path = payload["path"]
    github_links = get_github_links_from_issue_content(issue_content=issue_content)
    reference_file_contents = ""
    for link in github_links:
        file_content = get_github_file_content(owner=owner, repo=repo, path=link)
        reference_file_contents += file_content

    print(f"Received an issue event for {payload['repository']['name']}")

def handle_comment(payload):
    # TODO: Implement logic for handling comments
    openai.api_key = payload['openai_api_key']
    prompt = payload["prompt"]
    result = get_completion(prompt)

    print(f"Received a comment event for {payload['repository']['name']}")

def handle_pull_request(payload):
    # TODO: Implement logic for handling pull requests
    owner = payload["owner"]
    repo = payload["repo"]
    base_branch = payload["base_branch"]
    head_branch = payload["head_branch"]
    title = payload["title"]
    body = payload["body"]

    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "Bearer YOUR_ACCESS_TOKEN"
    }
    data = {
        "title": title,
        "body": body,
        "head": head_branch,
        "base": base_branch
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()

    if 'html_url' in response_json:
        pull_request_url = response_json['html_url']
        return pull_request_url
    else:
        return None
    print(f"Received a pull request event for {payload['repository']['name']}")
