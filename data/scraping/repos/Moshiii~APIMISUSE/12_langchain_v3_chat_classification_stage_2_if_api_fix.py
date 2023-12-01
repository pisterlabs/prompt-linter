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
# manual
# calib
# test_1
# test_2
# base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\calib\\"
base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\extra\\"
base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\manual\\"
# input_path = base_path + "calib_data_1k.json"
input_path = base_path + "manual_data_1k.json"
# input_path = base_path + "test_2_data_1k.json"
# input_path = base_path + "extra_data_223.json"
stage_1_path = base_path + "misuse_v2_stage_1_code_explain.json"
output_path = base_path + "misuse_v3_classification_stage_2_if_api_fix.json"

data_dict = {}
with open(input_path, encoding="utf-8") as f:
    data_manual = json.load(f)
    for line in data_manual:
        data_dict[line["number"]] = line
print("len(data_dict) ",len(data_dict))

data_stage_2_dict = {}
with open(stage_1_path, encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    data = [json.loads(line) for line in data]
    # data = data[:1000]
    print(len(data))
    for line in data:
        data_stage_2_dict[line["number"]] = line
print("len(data_stage_2_dict) ",len(data_stage_2_dict))


# merge data_stage_1_dict into data_dict by key
for key in data_dict.keys():
    if key in data_stage_2_dict.keys():
        data_dict[key]["code_change_explaination"] = data_stage_2_dict[key]["code_change_explaination"]
    else:
        print(key)
print("len(data_dict) ",len(data_dict))

# convert data_dict to list
data = []
for key in data_dict.keys():
    if "code_change_explaination" in data_dict[key].keys():
        data.append(data_dict[key])

print("len(data) ",len(data))

template_1 = """
You are an experienced software developer.
You excel at explaining code changes in a concise and easy-to-understand manner.
When you encounter a question to which you don't know the answer, you admit that you don't know.

Please read the definition of API misuse. These rules must be followed to answer the question correctly.
Next, read the context and the code change explanation.
Then, compare the removed code and added code to determine what has been changed.
If the code change involves only spacing, formatting, or comments, answer "No."
If the added code and removed code are almost identical, answer "No."
If the added code and removed code are not related to an API call, answer "No."
If the added code and removed code do not involve the use of machine learning-related APIs, answer "No."
Renaming variables does not constitute an API misuse fix, so answer "No."
Changing code formatting or style does not qualify as an API misuse fix, so answer "No."
Changing the type annotation in the form "def function(var: type)" does not indicate an API misuse fix, so answer "No."
Adding, removing, or changing the values or comments of variables or arrays does not indicate an API misuse fix, so answer "No."
Adding, removing, or changing imports does not signify an API misuse fix, so answer "No."
Adding, removing, or changing method definitions does not denote an API misuse fix, so answer "No."

Positive signs of API misuse include:
- When a developer uses an API in a way that is not intended by the API designer.
- API misuse fixes can involve deprecating APIs, fixing shape, type, dtype issues, or null reference checks.
- API misuse fixes can involve GPU, CPU, parallelization, or distributed fixes.
- API misuse fixes can involve using no_grad(), is_training, or eval() correctly, handling missing epsilon or atol, correcting the loss function, or fixing incorrect gradient calculations.
- API misuse fixes must include code changes related to API calls in the context section.
- API misuse fixes are highly likely to have API calls in the added code section.

Context:
{change_code}

Code Change Explanation:
{code_change_explanation}

Removed Code:
{removed_code}

Added Code:
{added_code}

<answer start>

Questions:
Is this an API misuse fix?

Answer: (Yes, No, Maybe)
"""


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(4))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)
# 154
for i in range(998, len(data)):

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

    code_change_explanation = data[i]["code_change_explaination"]
    print("code_change_explanation", len(code_change_explanation))
    prompt_1 = template_1.format(
        change_code=change, removed_code=removed, added_code=added, code_change_explanation=code_change_explanation)
    print("prompt_1", len(prompt_1))

    code_change_explanation = completion_with_backoff(
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


    code_change_explanation = code_change_explanation["choices"][0]["message"]["content"]

    output = {
        "number": number,
        "answer": code_change_explanation,
    }

    with open(output_path, 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
