import os
import openai
from git import Repo, InvalidGitRepositoryError
import tkinter as tk
from tkinter import simpledialog
import subprocess
import config

# Load OpenAI API key from config file
openai.api_key = config.OPENAI_API_KEY

# Set up Git repository URL and local path
git_repo_url = "https://github.com/OgeonX/ChatGPTPythonGIT.git"
local_repo_path = "C:\\Users\\admin\\source\\repos\\ChatGPTPythonGIT" # Change this to the path of your local repo

# Clone the Git repository if it doesn't exist locally, otherwise use the existing one
try:
    repo = Repo(local_repo_path)
except InvalidGitRepositoryError:
    # Read Git credentials from environment variables
    git_creds = os.environ.get("GIT_CREDENTIALS")
    if git_creds:
        # Clone the repository using Git credentials
        git_password = git_creds.split(":")[1]
        repo = Repo.clone_from(git_repo_url, local_repo_path, env={"GIT_ASKPASS": "git-gui--askpass"})
        repo.git.credentials.store("https://github.com", git_creds)
    else:
        # Clone the repository without Git credentials
        repo = Repo.clone_from(git_repo_url, local_repo_path)

# Function to read a file's content from the repository
def read_file(file_path):
    with open(file_path) as f:
        content = f.read()
    return content

while True:
    try:
        # Display a prompt for the user to enter a message
        root = tk.Tk()
        root.withdraw()
        user_input = simpledialog.askstring(title="Chat with OpenAI", prompt="Enter your message (type 'exit' to quit):")

        if user_input == "exit":
            break

        # Read the contents of all Python files in the repository, excluding the .git and __pycache__ folders
        files = [f for f in os.listdir(local_repo_path) if f.endswith(".py") and f not in [".git", "__pycache__", ".vs"]]
        file_contents = ""
        for file_name in files:
            file_path = os.path.join(local_repo_path, file_name)
            try:
                file_contents += read_file(file_path)
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")

            # Call OpenAI API to generate a response
            response = openai.Completion.create(
                engine="davinci",
                prompt="Context: {}\n\nUser Input: {}".format(file_contents, user_input),
                max_tokens=60,
                temperature=0.5,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n", " Human:", " AI:"],
                n=1,
            )
            generated_text = response.choices[0].text.strip()

            # Print the response
            print(generated_text)

            # Add the generated response to a new file in the repository
            file_name = "{}.txt".format(len(os.listdir(local_repo_path)) + 1)
            file_path = os.path.join(local_repo_path, file_name)
            with open(file_path, "w") as f:
                f.write(generated_text)

            # Determine the tag to use based on the file's status in the repository
            if repo.is_dirty(untracked_files=True):
                tag = "+"
            elif repo.untracked_files:
                tag = "-"
            else:
                tag = "*"

            # Commit the changes to the repository with the appropriate tag
            repo.git.add(".")
            repo.index.commit("{} {}".format(tag, file_name))

            # Push the changes to the remote repository
            p = subprocess.Popen(["git", "push"], cwd=local_repo_path)
            p.wait()

    except openai.error.APIError as e:
        print("There was an error with the OpenAI API: {}".format(e))

    except Exception as e:
        print("There was an error: {}".format(e))
    finally:
    # Close the Tk window
        root.destroy()
