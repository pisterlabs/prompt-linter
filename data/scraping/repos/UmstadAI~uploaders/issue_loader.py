import os
import requests
import time
import csv

from github import Github
from langchain.document_loaders import GitHubIssuesLoader

token = os.getenv('GITHUB_ACCESS_TOKEN') or 'GITHUB_ACCESS_TOKEN'

loader = GitHubIssuesLoader(
    repo="o1-labs/o1js",
    access_token=token,
    state="all",
    include_prs=False,
)

docs = loader.load()
issue_links = [link.metadata['url'] for link in docs]

def get_github_issue_and_comments(issue_link):
    # Extract the owner, repository, and issue number from the provided link
    parts = issue_link.strip('/').split('/')
    owner, repo, issue_number = parts[-4], parts[-3], parts[-1]

    g = Github(token)
    repo = g.get_repo(f"{owner}/{repo}")
    issue = repo.get_issue(int(issue_number))

    try:
        comments = issue.get_comments()
        comments_data = [comment.raw_data for comment in comments]
        return issue, comments
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return None, f"Error fetching comments: {e}"

def extract_issue_data(issue_links):
    issues = []
    x = 0
    for link in issue_links:
        issue_data, comments_data = get_github_issue_and_comments(link)
        if issue_data is None or comments_data is None:
            continue

        issue_number = issue_data.number
        issue_title = issue_data.title
        issue_writer = issue_data.user.login
        is_issue_open = issue_data.state
        issue_body = issue_data.body

        issue = {}

        comments = []
        for comment in comments_data:
            print(comment)
            comment_writer = comment.user.login
            comment_body = comment.body
            # comment_reactions = comment['reactions']
            # comment_reactions.pop('url', None)
            comments.append((comment_writer, comment_body))
        
        issue['number'] = issue_number
        issue['title'] = issue_title
        issue['writer'] = issue_writer
        issue['is_open'] = is_issue_open
        issue['body'] = issue_body
        issue['comments'] = comments

        issues.append(issue)
        print(issues)
        x += 1
        
    return issues

issue_data = extract_issue_data(issue_links)
print(len(issue_data))

file_name = 'output.csv'

with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)

def export_to_csv(data, file_name):
    field_names = data[0].keys()

    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=field_names)

        csv_writer.writeheader()

        for row in data:
            csv_writer.writerow(row)

export_to_csv(issue_data, file_name)