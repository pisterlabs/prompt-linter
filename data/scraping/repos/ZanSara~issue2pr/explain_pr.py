import os
import sys
import json
import subprocess
import openai

SYSTEM_PROMPT = lambda issue, diff: f"""
You just opened a PR to fix this issue:


{issue}


Your PR is the following:


{diff}


Write the body of the PR description. It should be technical, concise 
and contain a very brief description of what is the impact of your
PR on the codebase's behavior.

The PR description should help the maintainers
understand in detail your changes and review them.
It should also highlight any potentially unexpected behavior
that your changes will cause.

Use a professional tone. Never repeat yourself.
Be as clear and concise as possible.
Do not ask for the PR to be reviewed or merged: 
the contributors are already aware.
You must use Markdown to structure your reply.

Here is an example answer for an issue about divisions by zero:

    This PR makes the `divide` function fail with a helpful message
    when `divisor=0`, which looks like:

    ```
    InvalidDivisorError: no number can be divided by zero.
    ```

    Please make sure that the direct floats comparison I'm using
    does not cause floating point comparison issues.

Please output only the text of the PR description and nothing else.
"""

def explain_pr(issue_content, patch_path="changes.patch"):
    issue_data = json.loads(issue_content)
    patch = ""
    with open(patch_path, 'r') as patch_file:
        patch = patch_file.read()

    prompt = SYSTEM_PROMPT(f"# {issue_data['title']}\n\n{issue_data['body']}", patch)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    reply = response["choices"][0]["message"]["content"]
    
    print("#*********************")
    print(reply)
    print("#*********************") 
    
    reply = reply.replace('"', "\"")  # Bash
    return reply


if __name__ == "__main__":
    openai.api_key = sys.argv[1]
    print(explain_pr(issue_content=sys.argv[2]))
