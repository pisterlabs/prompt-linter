import git
import openai
import os
import json

def check_code_vulnerabilities(code, tokens):
    prompt = f"Tell me if there is a security vulnerability in the following code: '{code}' if there is a vulnerability in what line or lines and what you recommend to solve the problem with an example. Also, make the answer in Markdown format as if it was a GitHub comment"
    try:
        result = openai.Completion.create(
                    engine="text-davinci-003",
                    max_tokens=int(tokens),
                    top_p=1,
                    frequency_penalty=1,
                    presence_penalty=1,
                    prompt=prompt,
                    temperature=0.2
                    )
        if result["choices"][0]["text"].startswith("\n\nNo"):
            return "No vulnerabilities detected"
        else:
            return result["choices"][0]["text"]
    except Exception as e:
        return f"An error occurred: {e}"
    
def get_modified_files():
    try:
        with open(os.environ['GITHUB_EVENT_PATH']) as f:
            event = json.load(f)
        # Open the repository using the `git` library
        repo = git.Repo('.')

        # Get the base and head commit shas from the pull request event
        base_sha = event['pull_request']['base']['sha']
        head_sha = event['pull_request']['head']['sha']
        
        source_commit = repo.commit( base_sha )
        target_commit = repo.commit( head_sha )
              
        # Use the `git.Diff` class to get a list of modified files
        diff = source_commit.diff( target_commit )

        modified_files = []
        for d in diff:
            if d.change_type in ('A', 'M', 'T'):
                modified_files.append(d.a_path)
        return modified_files       
    except git.exc.InvalidGitRepositoryError as e:
        raise ValueError("The current directory is not a Git repository.") from e

def run():  # sourcery skip: avoid-builtin-shadow
    max = os.environ.get("MAX_FILES")
    tokens = os.environ.get("TOKENS")

    comment = "## Suggestions from the AI"

    files = get_modified_files()
    print(files)
    if len(files) > int(max):
        comment = f"{len(files)} files were modified. Limit {max}."
    else:
        for file in files:
            try:
                with open(file, 'r') as f:
                    code = f.read()
                iasuggestion = check_code_vulnerabilities(code, tokens)
            except Exception as e:
                print(f"An error occurred: {e}")
            comment = f"{comment}\n### {file}\n{iasuggestion}\n"
    with open("comment.md", "w") as f:
        f.write(comment)

if __name__ == "__main__":
    run()
