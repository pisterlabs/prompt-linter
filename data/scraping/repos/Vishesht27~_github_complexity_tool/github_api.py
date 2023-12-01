# github_api.py

import requests
import base64
import openai


def extract_username_from_url(url: str) -> str:
    """Extracts the username from a given GitHub URL."""
    return url.rstrip("/").split("/")[-1]


def fetch_repositories(username: str) -> list:
    """Fetches repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)

    if response.status_code == 200:
        return [repo["name"] for repo in response.json()]
    else:
        return []


def fetch_files_in_repository(username: str, repo_name: str) -> list:
    """Fetches a list of file paths in a given repository."""

    # First, fetch the default branch of the repository
    repo_info_url = f"https://api.github.com/repos/{username}/{repo_name}"
    repo_info_response = requests.get(repo_info_url)
    default_branch = repo_info_response.json().get("default_branch", "master")

    # Fetch the file tree for the default branch
    url = f"https://api.github.com/repos/{username}/{repo_name}/git/trees/{default_branch}?recursive=1"
    response = requests.get(url)

    if response.status_code == 200:
        tree = response.json().get("tree", [])
        return [
            item["path"] for item in tree if item["type"] == "blob"
        ]  # Filtering for files only
    else:
        return []


def fetch_file_content(username: str, repo_name: str, file_path: str) -> str:
    """Fetches the content of a specified file in the repository."""
    url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{file_path}"
    response = requests.get(url)

    if response.status_code == 200:
        content = response.json().get("content", "")
        # Content is base64 encoded, so we decode it
        return base64.b64decode(content).decode("utf-8")
    else:
        return ""




def evaluate_code_with_gpt3(code: str, api_key: str) -> float:
    """
    Uses GPT-3 to evaluate the technical complexity of a given code snippet.

    Parameters:
        - code (str): The code snippet to be evaluated.

    Returns:
        - float: A score representing the complexity of the code. The score is extracted from GPT-3's response.
    """
    # Crafting the prompt for GPT-3
    prompt_text = (
        "Please evaluate the technical complexity of the following code snippet "
        "on a scale of 1 to 10, where 1 is very simple and 10 is highly complex:\n\n"
        f"{code}\n\nComplexity Score (1-10): "
    )

    openai.api_key = api_key

    # Making a request to the OpenAI API
    response = openai.Completion.create(
        engine="davinci",  # Using the davinci engine for completion tasks
        prompt=prompt_text,
        max_tokens=50,  # Limiting the response length
    )

    # Extracting the complexity score from the response
    score_text = response.choices[0].text.strip()

    try:
        score = float(score_text)  # Convert the score to a float
        return min(max(score, 1), 10)  # Ensuring the score is between 1 and 10
    except ValueError:
        # If there's an issue parsing the score, return a default score or handle the exception as needed
        return 5.0