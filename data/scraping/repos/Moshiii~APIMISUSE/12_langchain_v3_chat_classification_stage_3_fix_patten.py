from dotenv import load_dotenv
import os
import json
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\manual\\"
input_path = base_path + "manual_data_1k.json"
stage_2_path = base_path + "misuse_v3_classification_stage_2.json"
stage_1_path = base_path + "misuse_v2_stage_1_code_explain.json"
output_path = base_path + "misuse_v3_classification_stage_3_fix_pattern.json"

# read
data_dict = {}
with open(input_path, encoding="utf-8") as f:
    data_manual = json.load(f)
    for line in data_manual:
        data_dict[line["number"]] = line



data_stage_1_dict = {}
with open(stage_1_path, encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    data = [json.loads(line) for line in data]
    print(len(data))
    for line in data:
        data_stage_1_dict[line["number"]] = line


data_stage_2_dict = {}
with open(stage_2_path, encoding="utf-8") as f:
    data = json.load(f)
    for line in data:
        data_stage_2_dict[line["number"]] = line

print(data_dict[0].keys())
print(data_stage_1_dict[0].keys())
print(data_stage_2_dict[0].keys())

# merge data_stage_1_dict into data_dict by key
for key in data_dict.keys():
    if key in data_stage_1_dict.keys():
        if key in data_stage_2_dict.keys():
            data_dict[key]["code_change_explaination"] = data_stage_1_dict[key]["code_change_explaination"]


data = []
for key in data_dict.keys():
    if "code_change_explaination" in data_dict[key].keys():
        data.append(data_dict[key])

# only keep these keys in data: number, change, commit_message, code_change_explaination
for i in range(0, len(data)):
    data[i] = {
        "number": data[i]["number"],
        "change": data[i]["change"],
        "added": data[i]["pos_line"],
        "removed": data[i]["neg_line"],
        "commit_message": data[i]["commit_message"],
        "code_change_explaination": data[i]["code_change_explaination"],
    }


print(len(data))
print(data[0].keys())
# exit()


template_1 = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Task: 
First please read the context, code change explaination, removed code, and added code.

Then, Please answer the following question and return the answer in given format at the end:
    what is the action of the fix?
    what is the API-element of the fix?

The following is the code change and similar example for your reference.
please limit your explaination to 3 sentences.

code change: 
{change_code}

removed code: 
{removed_code}

added code: 
{added_code}

code_change_explaination:
{code_change_explaination}


Please fill in the infomation in the given template below.

template:
action_of_fix: removal, addition, change, or update
API_element_of_fix: API call, API parameter, or API condition check
"""
# 1021
for i in range(0, len(data)):
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

    stage_3_answer = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt_1}
        ]
    )

    stage_3_answer = stage_3_answer["choices"][0]["message"]["content"]

    output = {
        "number": number,
        "stage_3_answer": stage_3_answer,
    }

    with open(output_path, 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
