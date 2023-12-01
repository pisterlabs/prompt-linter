import requests
import json
import base64
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv() 

llm = ChatOpenAI(temperature=0, model="gpt-4")


def bytes_to_base64(data):
    base64_data = base64.b64encode(data)
    return base64_data.decode('utf-8')


def get_branch_name(task):
    print('getting branch for ' + task)
    prompt_template = """
    You are an experienced developer chatbot. You are working on the below task:

    {task}

    Come up with a name for the GitHub branch for this pull request. Keep it short and to the point. Do not return anything other than the branch title itself.
    """

    llm_chain = LLMChain.from_string(llm=llm, template=prompt_template)
    answer = llm_chain.predict(task=task)
    return answer


def get_branch_sha(branch_name):

    repo_owner = os.environ["GITHUB_ORG"]
    repo_name = os.environ["GITHUB_REPO"]

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/refs"
    headers = {
        "Authorization": "Bearer " + os.environ['GITHUB_KEY']
    }
    response = requests.get(url, headers=headers)

    for branch in response.json():
        if branch['ref'] == f"refs/heads/{branch_name}":
            return branch['object']['sha']
    return None


def upload_files_to_github(title, task, branch_name, file_dict):

    repo_owner = os.environ["GITHUB_ORG"]
    repo_name = os.environ["GITHUB_REPO"]
    base_branch = os.environ["GITHUB_BRANCH"]

    base_branch_sha = get_branch_sha(base_branch)

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/refs"
    headers = {
        "Authorization": "Bearer " + os.environ['GITHUB_KEY']
    }

    branch_sha = get_branch_sha(branch_name)

    if not branch_sha:
        response = requests.post(url, headers=headers, data=json.dumps({"ref": f"refs/heads/{branch_name}", "sha": base_branch_sha}))
        print(response.text)
        if response.status_code == 201:
            print(f"Branch {branch_name} created successfully.")
            branch_sha = get_branch_sha(branch_name)
        else:
            print(f"Failed to create branch {branch_name}.")
            return

    for file_name in file_dict:
        base_64_file_content = bytes_to_base64(file_dict[file_name])
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_name}"

        data = requests.get(url+'?ref='+branch_name,
                            headers={"Authorization": "Bearer " + os.environ['GITHUB_KEY']
                                     }).json()

        # if base_64_file_content.decode('utf-8')+"\n" != data['content']:
        sha = data['sha']
        message = json.dumps({"message": "update",
                              "branch": branch_name,
                              "content": base_64_file_content,
                              "sha": sha
                              })
        resp = requests.put(url, data=message, headers={
            "Content-Type": "application/json",  "Authorization": "Bearer " + os.environ['GITHUB_KEY']
        })
        print(resp)



    data = {
        'title': title,
        'body': task['text'],
        'head': branch_name,
        'base': base_branch
    }

    print(data)
    response = requests.post(
        f'https://api.github.com/repos/{repo_owner}/{repo_name}/pulls', headers=headers, data=json.dumps(data))

    print(response.text)