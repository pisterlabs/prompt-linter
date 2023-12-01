from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from dotenv import load_dotenv
import os
import json
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\calib\\"
# base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\test_1\\"
# base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\test_2\\"
# base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\extra\\"
base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\manual\\"
# input_path = base_path + "test_1_data_1k.json"
input_path = base_path + "manual_data_1k.json"
# input_path = base_path + "extra_data_223.json"
# input_path = base_path + "calib_data_1k.json"
stage_1_path = base_path + "misuse_v2_stage_1_code_explain.json"
output_path = base_path + "misuse_v3_classification_stage_2_minor_change.json"

data_dict = {}
with open(input_path, encoding="utf-8") as f:
    data_manual = json.load(f)
    for line in data_manual:
        data_dict[line["number"]] = line

data_stage_2_dict = {}
with open(stage_1_path, encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    data = [json.loads(line) for line in data]
    # data = data[:1000]
    print(len(data))
    for line in data:
        data_stage_2_dict[line["number"]] = line


# merge data_stage_1_dict into data_dict by key
for key in data_dict.keys():
    if key in data_stage_2_dict.keys():
        data_dict[key]["code_change_explaination"] = data_stage_2_dict[key]["code_change_explaination"]

# convert data_dict to list
data = []
for key in data_dict.keys():
    if "code_change_explaination" in data_dict[key].keys():
        data.append(data_dict[key])

print(len(data))

template_1 = """
You are an experienced software developer.
You excel at explaining code changes in a concise and easy-to-understand manner.
When you encounter a question to which you don't know the answer, you admit that you don't know.

Please read the code change explanation, removed code, and added code. Then, answer the question at the end.

First, read the context and the code change explanation.
Next, compare the removed code and added code to determine what has been changed.
Then, determine if the code change is a minor change or not.

A source code minor change is a change that is not significant enough to be considered a major change. Minor changes include:
- Spacing changes
- Format changes
- Variable name changes
- String value changes
- Type annotation changes in the form "def function(var: type annotation)"
- Comment changes
- Added and removed lines that do not involve machine learning functions or methods

All other types of code changes are not considered minor changes.

Context:
{change_code}

Removed Code:
{removed_code}

Added Code:
{added_code}

<answer start>

Questions:
Is this code change a minor change?

Answer: (Yes, No, Maybe)
"""


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


# 844
for i in range(998, len(data)):

    # for i in range(110, 114):
    print("current_index:", i, "/", len(data))

    # commit_message = "{}\n".format(data[i]["commit_message"])
    change = ""
    removed = ""
    added = ""

    for j in range(0, len(data[i]["change"])):
        line = data[i]["change"][j]
        change += "{}\n".format(data[i]["change"][j])
        if line.startswith("-"):
            removed += "{}\n".format(data[i]["change"][j])
        if line.startswith("+"):
            added += "{}\n".format(data[i]["change"][j])
    number = data[i]["number"]
    print("number", number)
    # continue
    # print("commit_message", len(commit_message))
    print("change", len(change))
    print("removed", len(removed))
    print("added", len(added))

    code_change_explaination = data[i]["code_change_explaination"]
    print("code_change_explaination", len(code_change_explaination))
    prompt_1 = template_1.format(
        change_code=change, removed_code=removed, added_code=added, code_change_explaination=code_change_explaination)
    print("prompt_1", len(prompt_1))

    code_change_explaination = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt_1}
        ]
    )
    # code_change_explaination = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "user", "content": prompt_1}
    #     ]
    # )
    code_change_explaination = code_change_explaination["choices"][0]["message"]["content"]

    output = {
        "number": number,
        "answer": code_change_explaination,
    }

    with open(output_path, 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
