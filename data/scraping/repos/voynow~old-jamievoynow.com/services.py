import config
from dotenv import load_dotenv
from flask import url_for
from git2doc import loader
from llm_blocks import chat_utils
import openai
import os
import requests
import tiktoken
load_dotenv()


def fetch_portfolio():
    """Fetch pinned projects from GitHub"""
    endpoint = "https://api.github.com/graphql"
    headers = {"Authorization": f"Bearer {os.environ['GH_TOKEN']}"}

    response = requests.post(
        endpoint,
        json={"query": config.GH_GRAPHQL_QUERY % config.GITHUB_USERNAME},
        headers=headers,
    ).json()

    edges = response["data"]["user"]["pinnedItems"]["edges"]
    projects = [
        {
            "name": node["node"]["name"],
            "description": node["node"]["description"],
            "url": node["node"]["url"],
            "imageUrl": url_for('static', filename=f'assets/{node["node"]["name"]}.png'),
        }
        for node in edges
    ]
    return projects

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, allowed_special="all"))
    return num_tokens

def chat(query, project_name):
    """Chat with an LLM given the repo as context"""
    github_url = config.PROFILE_INFO["github"]
    repo_url = f"{github_url}/{project_name}"
    repo_docs = loader.pull_code_from_repo(repo_url)
    
    repo_str = ""
    for item in repo_docs:
        repo_str += f"{item['file_path']}:\n\n{item['page_content']}\n\n" 

    try:
        project_chat_chain = chat_utils.GenericChain(template=config.TEMPLATE, model_name="gpt-3.5-turbo-16k")
        response = project_chat_chain(repo_url=repo_url, repo=repo_str, query=query)["text"]
    except openai.error.InvalidRequestError:
        num_tokens = num_tokens_from_string(repo_str)
        response = f"I'm sorry, this repo is not supported yet due to context length limitations (currently attempting {num_tokens} tokens). We are actively working on fixing this!"

    return response
