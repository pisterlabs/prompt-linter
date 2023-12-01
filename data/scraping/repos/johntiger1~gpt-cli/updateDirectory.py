import os
import json
import mimetypes
import openai
import requests
import regex

def get_file_contents(filepath):
    mimetype, encoding = mimetypes.guess_type(filepath)
    if encoding is None:
        encoding = 'utf-8'
    with open(filepath, 'r', encoding=encoding) as f:
        try:
            return f.read()
        except UnicodeDecodeError:
            return None

def get_dir_contents(dirpath):
    dir_contents = {}
    for name in os.listdir(dirpath):
        path = os.path.join(dirpath, name)
        if os.path.isfile(path):
            contents = get_file_contents(path)
            if contents is not None:
                dir_contents[name] = contents
        elif os.path.isdir(path):
            sub_dir_contents = get_dir_contents(path)
            if sub_dir_contents:
                dir_contents[name] = sub_dir_contents
    return dir_contents

def dir_to_json(dirpath):
    dir_contents = get_dir_contents(dirpath)
    return json.dumps(dir_contents, indent=4)

def apply_changes_to_dir(changes, dirpath):
    for name, change in changes.items():
        path = os.path.join(dirpath, name)
        if isinstance(change, dict):
            if not os.path.isdir(path):
                os.mkdir(path)
            apply_changes(change, path)
        elif change == "__DELETE__":
            if os.path.isfile(path):
                os.remove(path)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(change)

# Prompt for OpenAI GPT-3.5-Turbo API
user_prompt = input("Please enter the prompt: ")

# Set up OpenAI API credentials
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set up the GPT-3 model
model_engine = "gpt-3.5-turbo"

with open('prompt.txt', 'r') as file:
    model_prompt = file.read().replace('\n', '')

gpt_input = [
    {"role": "system", "content": model_prompt}
    ]

# Prompt user to enter directory path
dirpath = input("Please enter the directory path: ")
json_data = dir_to_json(dirpath)

# Call OpenAI GPT-3.5-Turbo API to complete the JSON object
content = "User Prompt: " + user_prompt + "\n" + "JSON object representing directory and file content: " + "\n" + json_data

CONTENT_DICT = {"role": "user", "content": content}

gpt_input.append(CONTENT_DICT)

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=gpt_input
)

system_output_response = completion.choices[0].message.content
pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
json_string = pattern.findall(system_output_response)[0]
json_dict = json.loads(json_string)

print(json_dict)
# apply_changes_to_dir(dirpath, json_dict)