#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
from urllib.parse import urlparse

def extract_username(github_url):
    parsed_url = urlparse(github_url)
    path_segments = parsed_url.path.split("/")
    username = path_segments[1]  # Assuming the username is the second path segment
    return username

def fetch_user_repositories(github_url):
    username = extract_username(github_url)
    api_url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(api_url)
    repositories = response.json()
    return repositories


# In[10]:


username=input()
fetch_user_repositories(username)


# In[27]:


import openai
openai.api_key = 'Your-openai-key'
def generate_prompt(code):
    # Customize the prompt structure based on your requirements
    prompt = f"Code Complexity Analysis:\n\n```python\n{code}\n```\n\nEvaluate the technical complexity and challenges in the code."
    return prompt

def evaluate_code_complexity(code):
    prompt = generate_prompt(code)
    response = openai.Completion.create(
        engine='text-davinci-003',  # Replace with the appropriate GPT model
        prompt=prompt,
        max_tokens=200,  # Adjust token limit as needed
        temperature=0.7,
        n=1,
        stop=None
    )
    analysis = response.choices[0].text.strip()
    return analysis


# In[28]:


from github import Github

def fetch_user_repositories(github_url):
    # Extract the username from the GitHub URL
    username = extract_username(github_url)
    
    # Authenticate with a GitHub token if needed
    g = Github("Your github API")
    
    # Fetch the user object
    user = g.get_user(username)
    
    # Fetch the user's repositories
    repositories = []
    for repo in user.get_repos():
        repositories.append(repo)
    
    return repositories


def fetch_repository_code(repository):
    code = ""
    for file in repository.get_contents(""):
        if file.type == "file":
            try:
                # Fetch the code from the file
                code += file.decoded_content.decode(encoding="latin-1")  # Specify the appropriate encoding
            except (AssertionError, UnicodeDecodeError) as e:
                # Handle encoding errors
                print(f"Error fetching code for file {file.path}: {str(e)}")
        else:
            # Skip directories or other non-file types
            continue
    
    return code


# Example usage:
github_url = "https://github.com/venkatavinayvijjapu"
repositories = fetch_user_repositories(github_url)

for repository in repositories:
    code = fetch_repository_code(repository)
    print(f"Repository: {repository.full_name}")
    print(f"Code:\n{code}\n")
    analysis=evaluate_code_complexity(code)


# In[ ]:




