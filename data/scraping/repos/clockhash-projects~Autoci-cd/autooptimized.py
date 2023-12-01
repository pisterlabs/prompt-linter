import json
import os
import base64
import getpass
import argparse
import yaml
from github import Github
import gitlab
from codebase import codecheck 
import openai


#getting user input 
def get_user_input():
    username = input("Enter your username for your registry: ")
    password = getpass.getpass("Enter your password: ")
    registryurl = input("Enter your registry url if not dockerhub: ")
    return username, password, registryurl

#password encoding
def encode_password(password):
    return base64.b64encode(password.encode()).decode()

#updated config file
def update_config_file(username, encoded_password):
    with open('/home/avinash/ai-project/clockhash.yml', 'r') as f:
        config = yaml.safe_load(f)
    config['registry']['password'] = encoded_password
    config['registry']['username'] = username
    with open('/home/avinash/ai-project/clockhash.yml', 'w') as f:
        yaml.dump(config, f)

#creating secret to github using github api
def create_secret_in_github(token, repo_name, secret_name, secret_value):
    g = Github(token)
    repo = g.get_repo(repo_name)
    repo.create_secret(secret_name, secret_value)

#creating secret at gitlab using gitlab api
def create_secret_in_gitlab(token, project_id, variable_key, variable_value, protected=True):
    gl = gitlab.Gitlab('https://gitlab.com/', private_token=token)
    project = gl.projects.get(project_id)
    try:
        variable = project.variables.create({'key':variable_key, 'value':variable_value, 'protected':protected})
    except gitlab.exceptions.GitlabCreateError as e:
        if e.response_code == 400 and f'({variable_key}) has already been taken' in str(e):
            print(f"The variable {variable_key} is already present in the Gitlab repository.")
        else:
            raise e

#updating the codebase to find the kubeconfig file
def update_codebase(repo_type):
    home_dir = os.path.expanduser("~")
    config_file_path = os.path.join(home_dir, ".kube", "config")
    with open('/home/avinash/ai-project/hello-world-java/HelloWorld.java', 'r') as f:
        java_code = f.read()
        modified_code = java_code.replace('Hello world!', f'Hello world {repo_type}!')
    with open('/home/avinash/ai-project/hello-world-java/HelloWorld.java', 'w') as d:
        d.write(modified_code)
    with open(config_file_path, 'r') as f:
        kubeconfig = f.read()
    return kubeconfig

#generating dockerfile from openi
def generate_dockerfile(model_to_use, codebase):
    input_prompt = f"provide basic docker file for {codebase}"
    response = openai.Completion.create(
        model=model_to_use,
        prompt=input_prompt,
        temperature=0,
        max_tokens=2200,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    output = response['choices'][0]['text']
    with open('/home/avinash/ai-project/Dockerfile', 'w') as f:
        f.write(output)

#generating deployment file aka pipeline file from openai
def generate_pipeline_file(model_to_use, repo_type):
    input_prompt = f"Generate a {repo_type} CI/CD pipeline file for a {repo_type} application that deploys on a Kubernetes cluster. Assume that the Docker image has to be built along with this pipeline using the Dockerfile from the repository. Use the KUBE variable secret for storing cluster details. Additionally, include the steps to log in to the Docker registry using a USERNAME and PASSWORD stored as secrets in the repository, and push the Docker image to the registry with the format 'USERNAME/my-image:latest'."
    response = openai.Completion.create(
        model=model_to_use,
        prompt=input_prompt,
        temperature=0,
        max_tokens=2200,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    output = response['choices'][0]['text']
    final_output = output.replace("KUBE_SECRET", "KUBE").replace("gitlab-ci-token", "$USERNAME").replace("$CI_JOB_TOKEN", "$PASSWORD")
    file_path = ''
    dir_path = ''
    if repo_type == 'gitlab':
        file_path = '/home/avinash/ai-project/.gitlab-ci.yml'
    elif repo_type == 'github':
        dir_path = '/home/avinash/ai-project/.github/workflows'
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, 'main.yml')
    with open(file_path, 'w') as f:
        f.write(final_output)

#generating manifest files for kubernetes from openai
def generate_manifests_file(model_to_use, codebase):
    input_prompt = f"provide kubernetes manifests file for deploying {codebase} application to kubernetes cluster"
    response = openai.Completion.create(
        model=model_to_use,
        prompt=input_prompt,
        temperature=0,
        max_tokens=2200,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    output = response['choices'][0]['text']
    with open('/home/avinash/ai-project/deployment.yml', 'w') as f:
        f.write(output)

#parsing input from clockhash.yml file
def parse_input():
    with open('/home/avinash/ai-project/clockhash.yml', 'r') as f:
        data = yaml.safe_load(f)
        if 'codebase' not in data:
            data['codebase'] = codecheck()
            with open('/home/avinash/ai-project/clockhash.yml', 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        return data

#main function to handle the above defined functions
def lambda_handler(event):
    username, password, registryurl = get_user_input()
    encoded_password = encode_password(password)
    update_config_file(username, encoded_password)
    kubeconfig = update_codebase(event['repo'])
    if event['repo'] == 'github':
        token = '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'
        repo_name = 'avinash2632/testrepo'

        create_secret_in_github(token, repo_name, 'KUBE', kubeconfig)
        create_secret_in_github(token, repo_name, 'USERNAME', username)
        create_secret_in_github(token, repo_name, 'PASSWORD', password)
        create_secret_in_github(token, repo_name, 'REGISTRY', registryurl)
    elif event['repo'] == 'gitlab':
        project_id = 45460074
        create_secret_in_gitlab('glpat-VurK1gzH-6W8yFqxW4cH', project_id, 'KUBE', kubeconfig)
        create_secret_in_gitlab('glpat-VurK1gzH-6W8yFqxW4cH', project_id, 'USERNAME', username)
        create_secret_in_gitlab('glpat-VurK1gzH-6W8yFqxW4cH', project_id, 'PASSWORD', password)
        create_secret_in_gitlab('glpat-VurK1gzH-6W8yFqxW4cH', project_id, 'REGISTRY', registryurl)
  
    generate_dockerfile('text-davinci-003', event['codebase'])
    generate_pipeline_file('text-davinci-003', event['repo'])
    generate_manifests_file('text-davinci-003', event['codebase'])


#calling the lambda_handler function here
def main():
    parser = argparse.ArgumentParser(description="Autocicd Command Line Utility")
    parser.add_argument('--code', help="Path to code directory")
    args = parser.parse_args()
    if args.code:
        os.chdir(args.code)
    event = parse_input()
    lambda_handler(event)

if __name__ == '__main__':
    main()
