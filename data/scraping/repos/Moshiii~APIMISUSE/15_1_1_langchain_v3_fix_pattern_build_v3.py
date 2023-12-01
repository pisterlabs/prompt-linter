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
input_path = base_path + "manual_invest_data_1k.json"
output_path = base_path + "fix_rules_v3.json"

# read
data_dict = {}
with open(input_path, encoding="utf-8") as f:
    data_manual = json.load(f)
    for line in data_manual:
        data_dict[line["number"]] = line

data = []
for key in data_dict.keys():
    data.append(data_dict[key])
    # if data_dict[key]["label"] == "yes":
    #     data.append(data_dict[key])

# only keep these keys in data: number, change, commit_message, code_change_explaination
for i in range(0, len(data)):
    data[i] = {
        "change": data[i]["change"],
        "number": data[i]["number"],
        "label": data[i]["label"],
    }


print(len(data))
print(data[0].keys())


template_1 = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Task: 
please identify the trigger and fix  for Fixing API method problem in the following code change:

EXAMPLE:

code removed:
'''
"-        attention_scores = F.normalize(query_layer, dim=-1) @ F.normalize(key_layer, dim=-1).transpose(-2, -1)",
'''

code added:
'''
"+        attention_scores = nn.functional.normalize(query_layer, dim=-1) @ nn.functional.normalize(",
"+            key_layer, dim=-1",
"+        ).transpose(-2, -1)",
'''
answer:
<trigger>: F.normalize(
<replace>: nn.functional.normalize(



QUESTION:

code removed:
'''
{removed_code}
'''

code added:
'''
{added_code}
'''

<trigger>:
<replace>:
"""


# 48
for i in range(0, len(data)):
    print("current_index:", i, "/", len(data))
    change = ""
    code_before_change = ""
    code_after_change = ""
    context = ""
    added_code = ""
    removed_code = ""

    for j in range(0, len(data[i]["change"])):
        line = data[i]["change"][j]
        change += "{}\n".format(data[i]["change"][j])
        if line.startswith("+"):
            code_after_change += "{}\n".format(data[i]["change"][j][1:])
            added_code += "{}\n".format(data[i]["change"][j][1:])
        elif line.startswith("-"):
            code_before_change += "{}\n".format(data[i]["change"][j][1:])
            removed_code += "{}\n".format(data[i]["change"][j][1:])
        else:
            context += "{}\n".format(data[i]["change"][j])
            code_before_change += "{}\n".format(data[i]["change"][j])
            code_after_change += "{}\n".format(data[i]["change"][j])


    number = data[i]["number"]
    label = data[i]["label"]
    if label == "yes":
        if 
        print("number", number)
        print("change", len(change))

        prompt_1 = template_1.format(
            added_code=added_code, removed_code=removed_code
        )
        print("prompt_1", len(prompt_1))

        fix_pattern = completion_with_backoff(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt_1}
            ]
        )

        fix_pattern = fix_pattern["choices"][0]["message"]["content"]

        output = {
            "number": number,
            "change": change,
            "fix_pattern": fix_pattern,
        }

        with open(output_path, 'a') as f:
            json.dump(output, f)
            f.write(os.linesep)
