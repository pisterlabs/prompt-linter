"""
Git commit with automated commit message generation using OpenAI's GPT-3.

All the rights reserved:
Muktar SayedSaleh @ 2023
https://github.com/MukhtarSayedSaleh
https://www.linkedin.com/in/mukhtarsayedsaleh

"""

import subprocess
import openai
from dotenv import load_dotenv
import os

load_dotenv()


def get_git_diff(repo_path):
    """Get the git diff of the repo at the given path."""
    command = [
        "git", "-C", repo_path, "diff", "--staged"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout


def generate_commit_message(diff):
    """Generate a commit message using OpenAI's GPT-3."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model_id = os.getenv("OPENAI_MODEL_ID")

    prompt = """
        I have made the following changes to my git repository:\n{diff}\n,
        Please describe it for a git commit message, following the spec. below:\n
        The required format is as follows:
        a very short summary line, preferably around 50 chars, and strictly not more than 70 chars
        Do not invent anything that is not in the diff!
        Only describe the actual changes, lines starting with + and -, not the surrounding context.

        - concise info about first change, if needed, not more than 78 chars per line
        - concise info about second change, if any, not more than 78 chars per line
        - and so on

        Do not provide any reply besides the commit message itself.
    """

    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    # Assuming the last message from the assistant is the commit message
    commit_message = response["choices"][0]["message"]["content"]
    return commit_message


def git_commit(repo_path, message):
    """Commit the changes to the repo at the given path."""
    command = ["git", "-C", repo_path, "commit", "-m", message]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout


def run(repo_path):
    """
        Run the script.
        TODO: make it a python package and add to pypi.
    """
    diff = get_git_diff(repo_path)
    if diff:
        message = generate_commit_message(diff)
        message = message.replace("\"", "")
        result = git_commit(repo_path, message)
        print(result)
    else:
        print("No changes to commit.")


run(".")
