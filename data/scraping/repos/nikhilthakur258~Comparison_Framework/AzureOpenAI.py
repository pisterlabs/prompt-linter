import requests
import re
import openai
import json
import os

# Set up OpenAI API key
openai.api_key = '***'
openai.api_base = '****'
openai.api_type = 'azure'
openai.api_version = '2023-07-01-preview' # this may change in the future

deployment_name='OpenAIPOC'

# Set up Jira API credentials
jira_username = "nikhilthakur258@gmail.com"
jira_api_token = "***"
jira_issue_key = "GPTMVP-1"
jira_base_url = "***"
output_file = "Requirement_UI.txt"


# Function to get Jira issue description
def get_jira_description(issue_key):
    url = f"{jira_base_url}{issue_key}"
    auth = (jira_username, jira_api_token)
    response = requests.get(url, auth=auth)
    data = response.json()
    description = data['fields']['description']
    cleaned_description = re.sub(r'\{noformat}\|', '', description)
    cleaned_description = re.sub(r'\|\{noformat}', '', cleaned_description)
    description_to_append = "Write code in Java selenium to test below scenario: \n"
    description_to_append += cleaned_description

    # Append to file
    with open(output_file, 'w') as file:
        file.write(description_to_append)

    return cleaned_description

# Function to generate test cases using OpenAI GPT-3
def generate_test_cases(description_to_append):
    with open('Requirement_UI.txt') as f:
        content = f.read().replace('\n', ' ')
    prompt = content
    response = openai.Completion.create(engine=deployment_name, prompt=prompt,temperature=0.1,max_tokens=4000,top_p=0.95,frequency_penalty=0,presence_penalty=0,stop=None)
    for testcase in response["choices"]:
        print(f"{testcase['text']}")


# Main script
try:
    # Get Jira issue description
    jira_description = get_jira_description(jira_issue_key)

    # Generate test cases
    generated_test_cases = generate_test_cases(jira_description)


except Exception as e:
    print(f"An error occurred: {e}")


