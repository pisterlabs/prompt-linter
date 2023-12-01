import os
import re
from git import Repo
import openai
import yaml


# 1. Load config and Define the local directory
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

repo_URL = config['config']['repo_URL']
repo_name = repo_URL.split("/")[-1]
base_dir = os.getcwd()

# 2. Create new folder based on repo_name and clone the repo
source_repo_dir = os.path.join(base_dir, "repo_raw", repo_name)
target_repo_dir = os.path.join(base_dir, "repo_skated", repo_name)

# detect if the source repo already exists
if os.path.exists(target_repo_dir) and os.listdir(target_repo_dir):
    print('This repo already exists and we delete it for demo purpose.')
    import shutil
    shutil.rmtree(target_repo_dir)
else:
    os.makedirs(target_repo_dir, exist_ok=True)

Repo.clone_from(repo_URL, source_repo_dir)
print('Cloned the repo successfully. {} is created.'.format(target_repo_dir))

# 3. Change the working directory to the local directory and Pull the latest codebase from GitHub
os.chdir(target_repo_dir)
repo = Repo(target_repo_dir)
origin = repo.remotes.origin
origin.pull()

# %%

def extract_code_segments(text):
    code_segments = re.findall(r"```(?:Python|python)?\s*([\s\S]*?)```", text)
    if len(code_segments) == 0:
        raise Exception("No code segments found in the text.")
    return code_segments


def detect_openai_api(code_segment):
    if "openai" in code_segment.lower():
        return True
    else:
        return False


def update_file_with_llm(file_path):
    """

    :param file_path:
    :return -> bool: whether the file is updated.
    """
    # Load the codebase into a variable
    with open(file_path, 'r') as file:
        codebase = file.read()

    if not detect_openai_api(codebase):
        return False

    print("Updating the file: {}".format(file_path))

    respnose = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a Python code editor that reformat code, changing from OpenAI API to Langchain API."},
            {'role': 'user', "content": """This is usually how to call OpenAI API

```python
import openai

response = openai.ChatCompletion.create(
  model="gpt-4", # alternatively, you can use `model = 'gpt-3.5-turbo'`
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)

print(response['choices'][0]['message']['content'])
# print result: The 2020 World Series was played at the Globe Life Field in Arlington, Texas.
```

sample format for the response object:
```json
{
  "id": "chatcmpl-7k5fbq8QDmPg7vR9T0oHkAGl8aw85",
  "object": "chat.completion",
  "created": 1691219239,
  "model": "gpt-4-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The 2020 World Series was played at Globe Life Field in Arlington, Texas, USA."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 53,
    "completion_tokens": 19,
    "total_tokens": 72
  }
}
```

This is an example of calling chat models with OpenAI API in LangChain, performing the same utility. 

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# alternatively, you can use model_name = 'gpt-3.5-turbo'
llm = ChatOpenAI(model_name='gpt-4')

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Who won the world series in 2020?"),
    AIMessage(content="The Los Angeles Dodgers won the World Series in 2020."),
    HumanMessage(content="Where was it played?")
]
response = llm(messages)

# response: AIMessage(content='The 2020 World Series was played at Globe Life Field in Arlington, Texas.', additional_kwargs={}, example=False)
# to print the result, use `print(response.content)`
```

`{"role": "user", "content": "Who won the world series in 2020?"}` corresponds to  `HumanMessage(content="Who won the world series in 2020?")`
`{"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."}` corresponds to `AIMessage(content="The Los Angeles Dodgers won the World Series in 2020.")`

Both code snippets assume that there is an OPENAI_API_KEY environmental variable. If the key is set in other ways, make sure to set `os.environ["OPENAI_API_KEY"] = "whatever key format the codebase is using"`"""},
            {"role": "assistant", "content": "Understand, I will reformat a file from OpenAI API to LangChain. Please send me the file content to reformat."},
            {"role": "user", "content": """In the following code, please find usage of the OpenAI API, and replace it with LangChain API. Do not change anything else. 
\n```Python\n{code}\n```\nPlease output the complete file of the reformatted code and nothing else.""".format(code=codebase)},
        ]
    )

    output = respnose['choices'][0]['message']['content']

    cleaned_file_content = extract_code_segments(output)[0]

    # Update the codebase with the new code
    with open(file_path, 'w') as file:
        file.write(cleaned_file_content)

    return True


import tqdm
# 4. Load all .py files in local_dir separately and update the codebase with the new code
for root, dirs, files in tqdm.tqdm(os.walk(target_repo_dir)):
    # print(root, '===', dirs,'===', files)
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            is_updated = update_file_with_llm(file_path)
            if is_updated:
                print("Updated file: {}".format(file_path))


# 5. Commit the changes with a relevant message and push the changes to Github
repo.git.add(update=True)
repo.index.commit("Skateboard: Update codebase to LangChain.")

import random

# Generate a random integer between 0 and 999999
number = random.randint(0, 999999)

# Convert to string and pad with zeroes if necessary
random_string = str(number).zfill(6)

new_branch_name = 'Skateboard-LangChain_update'


# Create a new branch
repo.git.checkout('HEAD', b=new_branch_name)
# Push the new branch to the remote repository
repo.git.push('--set-upstream', 'origin', new_branch_name)

origin.push()
