# kudumisto
# GitHub Issue to GPT; GPT's Response as a Pull Request
#
# Code: https://github.com/busse/kodumisto
# Playground: https://github.com/busse/kodumisto-playground
# MIT License: https://github.com/busse/kodumisto/blob/main/LICENSE
#
# > Kodumisto (Esperanto): Derived from the Esperanto words 
# > "kodo" meaning "code" and "umisto" meaning "skilled person"
# > this name suggests the bot's expertise and competence in editing software code.
# >   -- ChatGPT-4
#
# Supports:
#
# NEW FILE: Write Issue to create new file. 
#  First line MUST be just a file extension including the '.'
#  The rest of the issue is the prompt to send to GPT.
#    
#  Examples:
#
#    Make a cool python script (issue title is ignored)
#    |  .py
#    |  Create a python script that writes "hello world" to the console.
#    |  It should also append one of ten random emojis to the end of the output.
#
#    Test data: The Simpsons did it first (issue title is ignored)
#    |  .csv
#    |  Create a csv file with sample transaction data for a credit card company.
#    |  The file should have a header row and fifty rows of data. The data should
#    |  represent credit card transactions for characters from the TV show The
#    |  Simpsons, mixed with transactions for characters from the TV show
#    |  Family Guy. Make the data similar, but similar transactions for The Simpsons
#    |  should have earlier timestamps than those for the Family Guy characters.
#
#    Write me a blog post (issue title is ignored)
#    |  .md
#    |  Write a post for my Jekyll / GitHub pages blog on what it means for AI to be
#    |  self-aware, but give wrong answers only.
#
#
# CHANGE FILE: Write Issue to create new file. 
#  First line of the Issue body MUST be the GitHub URL to the FILE to edit and return PR for.
#    Note: This is the same format that the GitHub Web UI uses when it creates an Issue from 
#    a file line number using "Reference in new issue"
#
#  Both github.com and raw.githubusercontent.com are supported
#
#  The rest of the issue is the prompt to send to GPT.
#    
#  Examples:
#
#    Let's go to Mars (issue title is ignored)
#    |  https://github.com/busse/kodumisto-playground/blob/main/directory1/hellomars.py
#    |  The output of this file should be the red planet, not world.
#
#    Test data: Migrate Legacy Simpsons data
#    |  https://github.com/busse/kodumisto-playground/blob/main/simpsons.csv
#    |  Refactor this csv file to represent The Family Guy instead of The Simpsons.
#
#    Add a new test to draw triangles (issue title is ignored)
#    |  https://raw.githubusercontent.com/busse/AI-Functions/master/test_ai_function.py
#    |  Add an additional test function to the script that generates a drawing of a triangle 
#    |  in SVG format with style infromation, and runs an appropriate test against it.
#    |  Take care to disturb what is already there as little as possible
#
#    TODO(busse): on the triangle test example above, I think it might not work with this latest
#                 version of kodumisto due to the outdated 'master' taxonomy, 
#                 but I have it working in an earlier dev version of this script.
#
# USAGE:
#  
#  kodumisto.py issue_number repo_owner repo_name [gpt_model defaults to gpt-3.5-turbo]
#
#  kodumisto.py 38 busse kodumisto-playground gpt-4
#

import os
import sys
import re
import time

from dotenv import load_dotenv

import requests

from github import Github
import openai

load_dotenv()

# Initialize variables
# Very likely there are better ways to handle this, and notably this will always cause cli to override env

if len(sys.argv) > 1: # TODO(busse): Need some error checking here
    argv_issue_number = sys.argv[1]

if len(sys.argv) > 2:
    repo_owner = sys.argv[2]
else:
    repo_owner = os.getenv("REPO_OWNER")

if len(sys.argv) > 3:
    repo_name = sys.argv[3]
else:
    repo_name = os.getenv("REPO_NAME")

if len(sys.argv) > 4:
    gpt_model = sys.argv[4]
else:
    gpt_model = "gpt-3.5-turbo"


random_string = int(time.time())
issue_number = argv_issue_number  # Set the issue number you want to read

# Set personal access tokens and repository details
openai_access_token = os.getenv("OPENAI_ACCESS_TOKEN")
github_access_token = os.getenv("GITHUB_ACCESS_TOKEN")

def get_filetype_prompt(string):
    # Find the index of the first newline character
    newline_index = string.find('\n')
    if newline_index == -1:
        # If there is no newline character, return the whole string as the first line
        return string, ''
    else:
        # Split the string at the newline character
        file_extension = string[:newline_index]
        prompt_body = string[newline_index+1:]
        return file_extension, prompt_body

def determine_file_action(issue_body):
    # Issues where the intent is to create a new file should start with the desired extension, ex:
    #  .py
    #  .csv
    #  .md
    # 
    # Note that the actual scripting or output formatting should still be specified in the Issue;
    #   The first line is not passed to the Prompt
    if issue_body.startswith("."):
        return "create"
    elif issue_body.startswith("http"):
        return "edit"
    else:
        return "create-no-extension" # TODO(): define how this should be handled

# For edits: The following three functions help parse the Issue body to get the file to edit
#   TODO(): refactor to one function?
def extract_url(text):
    regex = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(/[^\s]*)?'
    match = re.search(regex, text)
    if match:
        return match.group()
    return None

def convert_url_to_raw(text):
    strip_blob = text.replace("blob/", "")
    change_domain = strip_blob.replace("github.com", "raw.githubusercontent.com")
    return change_domain

def extract_filename(url, branch):
    parts = url.split("/")
    try:
        branch_index = parts.index(branch)
    except ValueError:
        return None

    path_parts = parts[branch_index + 1:]
    path = "/".join(path_parts)

    return path


def main():


    # Initialize the GitHub client
    gh = Github(github_access_token)
    repo = gh.get_repo(f"{repo_owner}/{repo_name}")

    # Construct the GitHub API URL
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}"

    # Set the request headers with the GitHub access token
    headers = {
        "Authorization": f"token {github_access_token}"
    }

    # Send a GET request to the GitHub API
    response = requests.get(api_url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        issue_data = response.json()
        issue_body = issue_data["body"]

        file_action = determine_file_action(issue_body)

        # If it's creating a new file, get the desired file type (for extension only) from the Issue

        if file_action == "create":
            file_extension, prompt_body = get_filetype_prompt(issue_body)

            # Write the remainder of the Issue as the Prompt
            prompt = f"{prompt_body}"

        # If it's an edit, we need to get the file path/name
        elif file_action == "edit":
            issue_branch = "main" # TODO(busse): Possible future logic to get files from branches other than main
            file_url = convert_url_to_raw(extract_url(issue_body))
            file_name = extract_filename(file_url, issue_branch)

            # Send a GET request to the GitHub API
            response_file = requests.get(file_url, headers=headers)

            # Check if the request was successful
            if response_file.status_code == 200:
                #file_data = f"{response_file.text}"
            
                #Set the prompt for an edit
                prompt = f"Ignoring any URLs here, apply these instructions: {issue_body}\n\nTo this code: {response_file.text}\n\nReturn only code in your response."
            else:
                print("Error fetching file:", response.status_code)
        
        else:
            print("Error fetching issue data:", response.status_code)


        # Initialize the OpenAI client
        openai.api_key = openai_access_token

        # Set the options for the OpenAI client
        model = gpt_model

        # Send the issue body as a request to ChatGPT
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{prompt}"}
            ]
        )


        chatgpt_response = response.choices[0].message.content
        #print("ChatGPT response:", chatgpt_response) # leaving this here for lazy debugging

        # Create a new branch
        branch_name = f"chatgpt-{model}-response-{issue_number}-{random_string}"
        base_branch = repo.get_branch("main")
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_branch.commit.sha)

        # If the Issue asks for a new file, create it
        if file_action == "create":
            # Create a new file with the ChatGPT response
            file_name = f"chatgpt-{model}-response-{issue_number}-{random_string}{file_extension}"
            repo.create_file(
                path=file_name,
                message=f"Add ChatGPT `{model}` response for issue #{issue_number}",
                content=chatgpt_response.encode("utf-8"),
                branch=branch_name,
            )

        elif file_action == "edit":
            #Get the file's content and SHA
            file_path = file_name
            file_contents = repo.get_contents(file_path, ref="main")
            file_sha = file_contents.sha

            # Update the file with the ChatGPT response
            new_file_data = f"{chatgpt_response}"
            repo.update_file(
                path=file_path,
                message=f"Update `{file_path}` with ChatGPT `{model}` response for issue {issue_number}",
                content=new_file_data.encode("utf-8"),
                sha=file_sha,
                branch=branch_name,
            )

        # Create a pull request
        pr_title = f"ChatGPT `{model}` response for issue #{issue_number}"
        pr_body = f"Pull request containing the ChatGPT `{model}` response for issue #{issue_number}:\n\n{chatgpt_response}"
        base = "main"
        head = branch_name
        pr = repo.create_pull(title=pr_title, body=pr_body, head=head, base=base)
        print(f"Successfully created pull request {pr.number}: {pr.title}")

    else:
        print("Error fetching issue data or error with GPT access:", response.status_code)

if __name__ == "__main__":
    main()


# MIT License
#
# Copyright (c) 2023 Chris Busse
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.