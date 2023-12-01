import requests
from termcolor import colored
from keys import OPENAI_API_KEY, GIT_TOKEN, GIT_USER_NAME
import openai
import json
import base64

def generate_documentation(data):

    openai.api_key = OPENAI_API_KEY

    command = 'Create a description for my project and a README.md. Return ONLY a JSON file with the data using the keywords description and readme. Feel free to add emojis.'

    command = f'{command}\n Project information: {data}'

    messages = [{"role": "system", "content": command}]

    completion = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo',
            messages = messages)
    
    response = completion['choices'][0]['message']['content']
    #print(response)
    output = json.loads(response)

    description = output['description']
    readme = output['readme']

    return description, readme

def create_repository():

    # GitHub API endpoint for creating repositories
    url = 'https://api.github.com/user/repos'

    # GitHub username and access token
    username = GIT_USER_NAME
    access_token = GIT_TOKEN

    # Repository name and description
    print(colored('\n\nSystem activated github module...', 'cyan'))
    repository_name =input(colored('\n- Enter your repository name:\n ', 'cyan'))
    data = input(colored('- Give me a short description about your project and I will create a fancy one:\n ', 'cyan'))
    

    description, readme = generate_documentation(data)
    # Request payload to create the repository
    repository_payload = {
        'name': repository_name,
        'description': description,
        'private': True  # Set to True for a private repository
    }

    # Set authentication headers
    headers = {
        'Authorization': f'token {access_token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    # Send the POST request to create the repository
    response = requests.post(url, json=repository_payload, headers=headers)

    # Check the response status code
    if response.status_code == 201:
        #print(f'Repository "{repository_name}" created successfully.')

        # Retrieve the repository details from the response
        repository_details = response.json()

        # Repository details
        owner = repository_details['owner']['login']
        repo = repository_details['name']

        # Path and name of the README.md file
        path = 'README.md'

        # Content of the README.md file
        readme_content_base64 = base64.b64encode(readme.encode("utf-8")).decode("utf-8")

        # Request payload to create or update the README.md file
        readme_payload = {
            'path': path,
            'message': 'Create or update README.md',
            'content': readme_content_base64
        }

        # Format the URL with repository details
        url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'

        # Send the PUT request to create or update the README.md file
        readme_response = requests.put(url, json=readme_payload, headers=headers)

        # Check the response status code for the README.md creation or update
        if readme_response.status_code == 201:
            return(f'{path} created or updated successfully.')
        else:
            return('Failed to create or update the README.md file.')
            #print(f'Response status code: {readme_response.status_code}')
            #print(f'Response body: {readme_response.text}')

    else:
        return('Failed to create the repository.')
        #print(f'Response status code: {response.status_code}')
        #print(f'Response body: {response.text}')

    
