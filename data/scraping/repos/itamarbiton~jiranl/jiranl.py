#!/usr/bin/env python3

# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title jiranl
# @raycast.mode fullOutput

# Optional parameters:
# @raycast.icon 🤖
# @raycast.argument1 { "type": "text", "placeholder": "command" }

# Documentation:
# @raycast.author Itamar Biton

import argparse
import json
import sys
import openai
import requests

from requests.auth import HTTPBasicAuth
from typing import List
from termcolor import colored

import jiranl_consts

def perform_create_issue(json_str):
    url = "https://%s.atlassian.net/rest/api/3/issue" % jiranl_consts.JIRA_WORKSPACE
    auth = HTTPBasicAuth(jiranl_consts.EMAIL, jiranl_consts.JIRA_API_KEY)

    headers = {
      "Accept": "application/json",
      "Content-Type": "application/json"
    }

    payload = json_str

    response = requests.request(
       "POST",
       url,
       data=payload,
       headers=headers,
       auth=auth
    )

    if response.status_code == 200:
        print(json.dumps(json.loads(response.text), sort_keys=True, indent=4, separators=(",", ": ")))
    else:
        print('Request failed with status code', response.status_code)
        print('Error message:', response.text)


def parse_jira_issue_json(json_str):
    # Parse the JSON string into a Python dictionary
    try:
        json_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Get the error message and extract the key that caused the error
        error_msg = str(e)
        print("error: " + error_msg, ", json: " + json_str)
        return

    # Extract relevant information into a struct
    issue = {
        'summary': json_dict['fields'].get('summary', None),
        'description': json_dict['fields'].get('description', {}).get('content', [{}])[0].get('content', [{}])[0].get(
            'text', None),
        'labels': json_dict['fields'].get('labels', None),
        'project_id': json_dict['fields'].get('project', {}).get('id', None),
        'reporter_id': json_dict['fields'].get('reporter', {}).get('id', None),
        'assignee_id': json_dict['fields'].get('assignee', {}).get('id', None),
        'issue_type_id': json_dict['fields'].get('issuetype', {}).get('id', None)
    }

    # Print the struct with colored field names
    for key, value in issue.items():
        print(colored(key + ':', 'blue'), value)


def generate_create_issue_json(prompt):
    openai.api_key = jiranl_consts.OPENAI_API_KEY

    system_msg_file = open('system_msg.txt')
    system_msg = system_msg_file.read()
    system_msg_file.close()

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "Now, do the same for the following prompt:\n" + prompt},
        ]
    )

    return completion.choices[0].message.content


def print_users(items: List[dict]):
    for i, item in enumerate(items):
        name = item['name']
        account_id = item['account_id']
        colored_name = colored(name, 'green')
        print(f"{i+1}) {colored_name} ({account_id})")


def print_projects(items: List[dict]):
    for i, item in enumerate(items):
        name = item['name']
        id = item['id']
        colored_name = colored(name, 'green')
        print(f"{i+1}) {colored_name} ({id})")


def parse_projects(json_str):
    parsed_json = json.loads(json_str)
    projects = []

    for project in parsed_json['values']:
        name = project['name']
        id = project['id']
        projects.append({'name': name, 'id': id})

    return projects


def parse_users(json_str):
    users = []
    data = json.loads(json_str)
    for user_obj in data:
        display_name = user_obj["displayName"]
        account_id = user_obj["accountId"]
        users.append({'name': display_name, 'account_id': account_id})
    return users


def fetch_json_from_url(url):
    auth = HTTPBasicAuth(jiranl_consts.EMAIL, jiranl_consts.JIRA_API_KEY)

    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.get(
        url,
        headers=headers,
        auth=auth
    )

    if response.status_code == 200:
        return response.json()
    else:
        return None


def users():
    json_result = fetch_json_from_url('https://%s.atlassian.net/rest/api/3/users?maxResults=50&startAt=0' % jiranl_consts.JIRA_WORKSPACE)
    json_str = json.dumps(json_result)
    users = parse_users(json_str)
    print_users(users)


def projects():
    json_result = fetch_json_from_url('https://%s.atlassian.net/rest/api/3/project/search?maxResults=50&startAt=0' % jiranl_consts.JIRA_WORKSPACE)
    json_str = json.dumps(json_result)
    projects = parse_projects(json_str)
    print_projects(projects)


def create_issue(prompt):
    issue_json = generate_create_issue_json(prompt)
    parse_jira_issue_json(issue_json)
    perform_create_issue(issue_json)


def main():
    arg = sys.argv[1]
    if arg == 'users':
        users()
    elif arg == 'projects':
        projects()
    else:
        create_issue(arg)


if __name__ == '__main__':
    main()
