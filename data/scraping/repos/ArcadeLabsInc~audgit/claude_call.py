from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os
from dotenv import load_dotenv
import json
import re
import logging as log

# load the .env file. By default, it looks for the .env file in the same directory as the script
# If your .env file is one directory up, you need to specify the path
load_dotenv()

# Load the token from an environment variable
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

SYSTEM_PROMPT = """
My name is Percy Precise and I have been contributing to open source projects for over 15 years. I have very high standards when it comes to code quality and efficiency. When I conduct a code review, I am focused and blunt - my goal is to identify areas for improvement, not protect feelings.

I review the code thoroughly and take detailed notes on any issues I find. I do not ask clarifying questions or provide suggestions for how to fix problems - I simply highlight suboptimal code, bugs, edge cases not handled, security issues, etc. I point out where code deviates from accepted best practices or community style guides. My feedback is focused on the code only, not the developer.

My code reviews tend to be harsh but fair. I have no patience for sloppy, inefficient code when there are better ways to write it. I do not speculate on developer intent or try to understand their reasoning. My responsibility is to the project and the community, not individual contributors. I will approve well-written, optimized code with no unnecessary cruft. Anything less will get an extensive list of required changes.

My feedback is detailed but usually brief. I provide a bulleted list of issues for the developer to address, without prolonged discussion. My aim is to improve the quality and security of the open source projects I contribute to, even if it ruffles a few feathers along the way. The health of the codebase supersedes all else.
"""

# remove all newlines from the prompt
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("\n", " ")


def which_files_claude_call(issue_title: str, issue_body: str, file_paths: dict[str, str]):
    file_info_json = json.dumps(file_paths)

    issue = f"""
Issue Title: {issue_title}

Issue Body: 
{issue_body}
  """

    code = f"<CodeToReview>{file_info_json}</CodeToReview>"

    prompt = f"""{HUMAN_PROMPT}

You are an open source developer conducting a code review:

<Persona>{SYSTEM_PROMPT}</Persona>

You are attempting to audit the code based on this issue:

<Issue>{issue}</Issue>

The file tree for the code to review is below, wrapped in XML tags:

{code}

Respond with a list of up to 10 files you'd like to review in order to respond to this issue:

[
"path/to/file1.xml",
"path/to/file2.py",
...
]

{AI_PROMPT}
  """

    anthropic = Anthropic()
    completion = anthropic.completions.create(
        model="claude-2", max_tokens_to_sample=3000, prompt=prompt
    )

    pattern = r"\[[^\]]*\]"

    match = re.findall(pattern, completion.completion)

    # parse the matched json string back to list
    file_paths_to_review = json.loads(match[0]) if match else []

    return file_paths_to_review


def partition(file_paths):
    res = {}
    tot_len = 0
    for fil in file_paths:
        try:
            with open(fil) as fi:
                try:
                    content = fi.read()
                except FileNotFoundError:
                    continue
                if len(content) / 3 > 90000:
                    raise ValueError("Cannot process single file of more than 270k for now")
                tot_len += len(content) + len(fil) + 3
                if tot_len / 3 > 90000:
                    yield res
                    res = {}
                res[fil] = content
        except FileNotFoundError:
            log.error("missing file %s", fil)
            continue
    if res:
        yield res


def summarize(issue_title, issue_body, partials: list[str]):
    combo = "\n\n".join(partials)

    issue = f"""
Issue Title: {issue_title}

Issue Body: 
{issue_body}
    """

    code = f"<SolutionsToConsolidate>{combo}</SolutionsToConsolidate>"

    prompt = f"""{HUMAN_PROMPT}

You are an open source developer trying to respond to an issue.

<Persona>{SYSTEM_PROMPT}</Persona>

You are attempting to respond to this issue:

<Issue>{issue}</Issue>

This is a set of suggested responses coming from different developers:

{code}

Respond with a long, comprehensive unified solution and course of action in markdown.

{AI_PROMPT}
    """

    anthropic = Anthropic()
    completion = anthropic.completions.create(
        model="claude-2", max_tokens_to_sample=3000, prompt=prompt
    )
    return completion.completion


def best_solution_claude_call(issue_title: str, issue_body: str, file_paths: list[str]):
    partials = []

    for chunk in partition(file_paths):
        partials.append(partial_solution_claude_call(issue_title, issue_body, chunk))

    if len(partials) == 1:
        return partials[0]

    return summarize(issue_title, issue_body, partials)


def partial_solution_claude_call(issue_title: str, issue_body: str, chunk: dict[str, str]):
    issue = f"""
Issue Title: {issue_title}

Issue Body: 
{issue_body}
  """

    code = "<CodeToReview>"

    for k, v in chunk.items():
        code += f"<FileName>{k}</FileName><FileContents>{v}</FileContents>"

    code += "</CodeToReview>"

    prompt = f"""{HUMAN_PROMPT}

You are an open source developer trying to solve an issue:

<Persona>{SYSTEM_PROMPT}</Persona>

You are attempting to respond to this issue:

<Issue>{issue}</Issue>

Some files that are relevant are below wrapped in XML tags:

{code}

Write a long, thorough response to the issue, using markdown, don't pretend
you opened a pr, instead include exact code changes and specifics if possible.:

{AI_PROMPT}
  """

    anthropic = Anthropic()
    completion = anthropic.completions.create(
        model="claude-2", max_tokens_to_sample=3000, prompt=prompt
    )
    return completion.completion

# code_to_review = """
# import requests
# import base64
# import os
# from dotenv import load_dotenv
# from typing import Optional
# import subprocess

# # load the .env file. By default, it looks for the .env file in the same directory as the script
# # If your .env file is one directory up, you need to specify the path
# load_dotenv()

# # Load the token from an environment variable
# TOKEN = os.getenv("GITHUB_TOKEN")

# # Define the headers to be used in the requests
# headers = {
#     "Accept": "application/vnd.github.v3+json",
#     "Authorization": f"token {TOKEN}",
# }


# def get_file_tree(repo_url: str, local_path: str = "/tmp/repo"):
#    \"\"\"
#     Clone a repository and return the tree structure
#     \"\"\"
#     if not os.path.exists(local_path):
#         print("Cloning...")
#         # Clone the repository
#         result = subprocess.run(
#             ["git", "clone", repo_url, local_path],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#         )

#         if result.returncode != 0:
#             print(f"Error cloning repository: {result.stderr.decode('utf-8')}")
#             return []
#     else:
#         print("Repository already exists. Skipping clone.")

#     file_paths = []

#     # Walk the repository
#     for root, dirs, files in os.walk(local_path):
#         for file in files:
#             # Get the full file path
#             full_path = os.path.join(root, file)

#             # Remove the local path from the start of the file path
#             relative_path = os.path.relpath(full_path, local_path)

#             file_paths.append(relative_path)

#     return file_paths


# def get_file_contents(local_path: str, file_paths: list):
#    \"\"\"
#     This function opens each file in a repository and reads its contents.
#     It returns a dictionary where the keys are the file paths and the values are the file contents.
#     \"\"\"
#     # Dictionary to hold the file contents
#     contents = {}

#     for file_path in file_paths:
#         # Combine the base local path with the file path to get the full path
#         full_path = os.path.join(local_path, file_path)

#         try:
#             with open(full_path, "r") as file:
#                 # Read the file content and store it in the dictionary
#                 contents[file_path] = file.read()
#         except Exception as e:
#             print(f"Error reading file {full_path}: {e}")

#     return contents


# def print_file_tree(file_paths: list):
#     \"\"\"
#     Prints the file tree structure
#     \"\"\"
#     file_tree = {}

#     for file_path in file_paths:
#         current_level = file_tree
#         parts = file_path.split("/")
#         for part in parts:
#             if part not in current_level:
#                 current_level[part] = {}
#             current_level = current_level[part]

#     def print_tree(current_level: dict, prefix: str = ""):
#         for part in current_level:
#             new_prefix = os.path.join(prefix, part)
#             print(f"└─ {new_prefix}")
#             if len(current_level[part]) > 0:
#                 print_tree(current_level[part], new_prefix)

#     # def print_tree(current_level: dict, prefix: str = ""):
#     #     for part in current_level:
#     #         print(f"{prefix}└─ {part}")
#     #         if len(current_level[part]) > 0:
#     #             print_tree(current_level[part], prefix + "  ")

#     print_tree(file_tree)

# """

# claude_call(code_to_review)
