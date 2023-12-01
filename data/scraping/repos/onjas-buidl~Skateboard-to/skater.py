import os
import re
from git import Repo
import openai
import yaml
import tqdm

# 1. Load config and Define the local directory
USE_SEPARATE_REPO = True
AUTOMATIC_SEARCH_FILE_TO_UPDATE = True


if USE_SEPARATE_REPO:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    target_repo_dir = config['config']['repo_path']  # '/Users/jasonhu/Desktop/GPT_email_generator'
else:
    repo_name = 'GPT_email_generator'  # repo_URL.split("/")[-1]
    base_dir = os.getcwd()
    target_repo_dir = os.path.join(base_dir, "repo_skated", repo_name)

os.chdir(target_repo_dir)
repo = Repo(target_repo_dir)


# %%

def extract_code_segments(text):
    code_segments = re.findall(r"```(?:Python|python)?\s*([\s\S]*?)```", text)
    if len(code_segments) == 0:
        raise Exception("No code segments found in the text.")
    return code_segments


def detect_openai_api(code_segment):
    if "openai" in code_segment:
        return True
    else:
        return False


def update_file_with_llm(file_content):
    """

    :param file_path:
    :return -> bool: whether the file is updated.
    """
    # Load the codebase into a variable
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        messages=[
            {"role": "system",
             "content": "You are a Python code editor that reformat code, changing from OpenAI API to Langchain API."},
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
            {"role": "assistant",
             "content": "Understand, I will reformat a file from OpenAI API to LangChain. Please send me the file content to reformat."},
            {"role": "user", "content": """In the following code, please find usage of the OpenAI API, and replace it with LangChain API. Do not change anything else. 
\n```Python\n{code}\n```\nPlease output the complete file of the reformatted code and nothing else.""".format(
                code=file_content)},
        ]
    )

    output = response['choices'][0]['message']['content']

    updated_file_content = extract_code_segments(output)[0]



    return updated_file_content


AUTOMATIC_SEARCH_FILE_TO_UPDATE=False
if AUTOMATIC_SEARCH_FILE_TO_UPDATE:
    for root, dirs, files in tqdm.tqdm(os.walk(target_repo_dir)):
        # print(root, '===', dirs,'===', files)
        for file in files:
            if not file.endswith('.py'):
                continue
            with open(os.path.join(root, file), 'r') as file:
                file_content = file.read()
            if detect_openai_api(file_content):
                print("Found OpenAI API in file: {}".format(file))
                update_file_with_llm(file_content)
                print("Updated file: {}".format(file))
else:
    files_paths_to_update = ['eval/score.py']
    for file_path in tqdm.tqdm(files_paths_to_update):
        with open(file_path, 'r') as file:
            file_content = file.read()
        updated_file_content = update_file_with_llm(file_content)
        # Update the codebase with the new code
        with open(file_path, 'w') as file:
            file.write(updated_file_content)
        print("Updated file: {}".format(file_path))
