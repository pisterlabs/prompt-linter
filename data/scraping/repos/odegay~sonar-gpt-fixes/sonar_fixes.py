import os
import requests
import json
import openai
import base64
import sys
import subprocess
import tempfile
from git import Repo
from datetime import datetime

SONAR_API_URL = "https://sonarcloud.io/api"
# PROD CONSTANTS
# SONAR_ORG_KEY = os.environ["SONAR_ORGANIZATION_KEY"]
# SONAR_PROJECT_KEY = os.environ["SONAR_PROJECT_KEY"]
# SONAR_TOKEN = os.environ["SONAR_TOKEN"]
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# GITHUB_OWNER = os.environ["GITHUB_OWNER_ENV"]
# GITHUB_REPO_NAME = os.environ["GITHUB_REPO_NAME_ENV"]
# GITHUB_ACCESS_TOKEN = "os.environ['GITHUB_ACCESS_TOKEN_ENV']"
# MAX_CYCLES = int(os.environ.get("MAX_CYCLES", 3))  # Default value is 3 cycles

# DEV CONSTANTS


def fetch_issues(sonar_token, source_directory):
    SONAR_API_BASE_URL = "https://sonarcloud.io/api"
    ORGANIZATION_KEY = SONAR_ORG_KEY
    PROJECT_KEY = SONAR_PROJECT_KEY

    auth_header = base64.b64encode(f"{sonar_token}:".encode()).decode()

    # Define the function to fetch issues from the SonarCloud API
    def fetch_paged_issues(page_index):
        try:
            response = requests.get(
                f"{SONAR_API_BASE_URL}/issues/search",
                params={
                    "organization": ORGANIZATION_KEY,
                    "projects": PROJECT_KEY,
                    "types": "CODE_SMELL, BUG, VULNERABILITY",
                    "statuses": "OPEN, CONFIRMED, REOPENED",
                    "p": page_index,
                },
                headers={"Authorization": f"Basic {auth_header}"},
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            raise Exception(f"Error: Failed to fetch issues from SonarCloud API: {str(e)}")

    # Fetch all pages of issues
    issues_by_file = {}
    page_index = 1
    while True:
        try:
            result = fetch_paged_issues(page_index)
        except Exception as e:
            print(e)
            sys.exit(1)

        issues = result["issues"]

        if not issues:
            break

        for issue in issues:
            # Remove the project key from the component
            file_path = issue["component"].replace(SONAR_PROJECT_KEY + ":", "")
            file_path = os.path.join(source_directory, file_path)
            line = issue.get("line", 0)
            message = issue["message"]

            if file_path not in issues_by_file:
                issues_by_file[file_path] = []

            issues_by_file[file_path].append({
                "line": line,
                "message": message,
        })

        page_index += 1

    return issues_by_file

def generate_prompt(file_content, issue):
    return f"##### The SonarCloud found the following issue on line {issue['line']}: {issue['message']}\n \n### Code with issues\n{file_content}\n \n### Fixed Code that only contains fixed block of lines of code and not the entire code:"

def apply_suggested_fix(file_content, issue, suggested_fix):
    lines = file_content.split('\n')
    issue_line = issue['line'] - 1
    suggested_lines = suggested_fix.split('\n')

    # Replace the affected lines with the suggested fix lines
    lines[issue_line : issue_line + len(suggested_lines)] = suggested_lines

    return '\n'.join(lines)

def implement_fixes(issues_by_file):
    openai.api_key = OPENAI_API_KEY

    for file_path, issues in issues_by_file.items():
        # Read the file contents
        with open(file_path, 'r') as file:
            file_content = file.read()

        # Sort the issues in descending order by line number
        issues.sort(key=lambda x: x['line'], reverse=True)

        # Iterate through issues in the file
        for issue in issues:
            # Generate the prompt using the current file_content, which includes previously applied fixes
            prompt = generate_prompt(file_content, issue)
            print(f"Generating suggestion for the following prompt: {prompt}")
            try:
                response = openai.Completion.create(
                    model="text-davinci-003",                    
                    prompt=prompt,
                    temperature=0,
                    max_tokens=182,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=["###"]
                    )
                suggested_fix = response.choices[0].text.strip()
                print(f"Suggested fix for issue '{issue}': {response}")
            except Exception as e:
                print(f"Error: Failed to get a suggestion from GPT-4 for issue '{issue}': {str(e)}")
                continue

            # Apply the suggested fix to the file content
            file_content = apply_suggested_fix(file_content, issue, suggested_fix)

        # Write the updated file content back to the file
        with open(file_path, 'w') as file:
            file.write(file_content)
            print(f"Updated file: {file_path}")

def create_pr(base, head, title):
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO_NAME}/pulls"
    headers = {"Authorization": f"token {GITHUB_ACCESS_TOKEN}"}
    data = {
        "title": title,
        "head": head,
        "base": base,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    return response.json()

def main():   

    # Create a temporary directory to clone the repo
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Clone the repo
        repo_url = f'https://{GITHUB_ACCESS_TOKEN}@github.com/{GITHUB_OWNER}/{GITHUB_REPO_NAME}.git'
        repo = Repo.clone_from(repo_url, tmp_dir, branch='master')

        # Set author identity for Git
        repo.config_writer().set_value("user", "name", "odegay").release()
        repo.config_writer().set_value("user", "email", "dolegan@gmail.com").release()


        # Fetch issues from the SonarCloud API
        try:
            issues_by_file = fetch_issues(SONAR_TOKEN, tmp_dir)
        except Exception as e:
            print(f"Error: Failed to fetch issues from SonarCloud API: {str(e)}")
            sys.exit(1)

        # Create a new branch for the fixes with a unique name
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        new_branch_name = f"fixes_{current_time}"
        new_branch = repo.create_head(new_branch_name)

        # Checkout the new branch
        new_branch.checkout()

        # Implement fixes using the GPT-4 API
        try:
            implement_fixes(issues_by_file)
        except Exception as e:
            print(f"Error: Failed to implement fixes using GPT-4 API: {str(e)}")
            sys.exit(1)

        # Commit the changes
        repo.git.add(A=True)
        repo.git.commit(m='Apply automated fixes')

        # Push the changes to the new branch
        repo.git.push('--set-upstream', 'origin', new_branch.name)

    # Create a PR
    try:
        pr = create_pr('master', new_branch.name, 'Apply automated fixes')
        print(f"Created PR: {pr['html_url']}")
    except requests.RequestException as e:
        print(f"Error: Failed to create PR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()