import os
import json
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import requests
import torch
import tiktoken
import argparse


commit_schema = {
    "name": "git_commit",
    "description": 'Performs a git commit by calling `git commit -m "commit_message"`',
    "parameters": {
        "type": "object",
        "properties": {
            "commit_message": {
                "description": "A short but descriptive commit message",
                "type": "string"
            }
        },
        "required": ["commit_message"]
    }
}

def generate_commit_message_mistral(diff):
    """Generate commit message using Mistral AI."""
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokens = tokenizer.encode(diff)
    tokens = tokens[:7999]
    diff = tokenizer.decode(tokens)
    prompt = "You are given the output of a git diff. Your task is to create a descriptive commit message based on this diff, max 15 words\n\n" + diff
    data = {
        "system": "You generate commit messages from a git diff that is provided to you. It is your job to create a descriptive commit message based on this diff. Do not include the diff in your commit message. Only include the commit message. The most important thing is to ensure you are only describing the changes that are marked with + or - in the diff. Do not include any other changes in your commit message.",
        "model": "mistral",
        "prompt": "{prompt}".format(prompt=prompt),
        "stream": False,
    }
    response = requests.post("http://localhost:11434/api/generate", json=data)
    json_strings = response.text.strip().split('\n')
    responses = [json.loads(js)["response"] for js in json_strings]
    result = "".join(responses)

    return result
    
def generate_commit_message_globe_server(diff):
    data = {"diff": diff}
    response = requests.post("http://globe.engineer/api/scommit-server", json=data)
    commit_message = response.text.strip()
    return commit_message

def format_diff(diff):
    added = []
    removed = []
    lines = diff.split('\n')
    for line in lines:
        if line.startswith('+'):
            added.append(line)
        elif line.startswith('-'):
            removed.append(line)
    formatted_diff = 'ADDED:\n' + '\n'.join(added) + '\nREMOVED:\n' + '\n'.join(removed)
    return formatted_diff

def generate_commit_message_gpt(diff):
    """Generate commit message using OpenAI's ChatGPT."""

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

    if len(diff) == 0:
        return 'default commit message'

    tokens = tokenizer.encode(diff)
    tokens = tokens[:15900]
    diff = tokenizer.decode(tokens)
    prompt = "Can you commit this diff for me:\n\n" + diff

    response = client.chat.completions.create(messages=[
        {'role': 'system', 'content': "You call the git commit function with short and informative commit messages"},
        {'role': 'user', 'content': prompt},
    ],
    functions=[commit_schema],
    function_call={'name': 'git_commit'},
    model='gpt-3.5-turbo-16k',
    temperature=0.5)
    args = json.loads(response.choices[0].message.function_call.arguments)
    commit_message = args['commit_message']
    return commit_message


def scommit():
    """Perform a git commit with a generated or provided message."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, help='Commit message')
    parser.add_argument('-mi', action='store_true', help='Using mistral')
    parser.add_argument('-globe-server', action='store_true', help='Using globe server')
    args, unknown = parser.parse_known_args()

    try:
        # Check if there are any commits
        subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD'], text=True).strip()
        commits_exist = True 
    except subprocess.CalledProcessError:
        commits_exist = False

    if commits_exist and args.mi:
        diff = subprocess.check_output(['git', 'diff', 'HEAD'] + unknown, text=True).strip()
        formatted_diff = format_diff(diff)
        message = generate_commit_message_mistral(formatted_diff)
        message = message.replace('"', '\\"')
    
    elif commits_exist and args.globe_server:
        diff = subprocess.check_output(['git', 'diff', 'HEAD'] + unknown, text=True).strip()
        formatted_diff = format_diff(diff)
        message = generate_commit_message_globe_server(formatted_diff)
        message = message.replace('"', '\\"')
    
    elif args.m is None and commits_exist:
        diff = subprocess.check_output(['git', 'diff', 'HEAD'] + unknown, text=True).strip()
        formatted_diff = format_diff(diff)
        message = generate_commit_message_gpt(formatted_diff)

    else:
        message = args.m if args.m is not None else 'Initial commit'

    cmd = f'git commit {" ".join(unknown)} -m "{message}"'
    os.system(cmd)
    

if __name__ == '__main__':
    scommit()