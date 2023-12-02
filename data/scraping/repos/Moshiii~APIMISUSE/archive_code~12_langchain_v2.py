
from dotenv import load_dotenv
import os
import json
import openai
load_dotenv()

# read
with open("C:\@code\APIMISUSE\data\misuse_jsons\manual\merged_split_hunk_AST_filter_manual_deduplica_reduced_category.json", encoding="utf-8") as f:
    data = json.load(f)
# get commit message and change
openai.api_key = os.getenv("OPENAI_API_KEY")

template_1 = """
You are an experienced software developer.
You are great at explain code changes in a concise and easy to understand manner.
When you don't know the answer to a question you admit that you don't know.

Please provide read the following code commit and provide a code change explaination.
Please including the following points in your code change explaination:
    The motivation of the code change.
    The solution to the code change.
please limit your explaination to 3 sentences.

code change: 
{change_code}

removed code: 
{removed_code}

added code: 
{added_code}

code change explaination:
"""

template_2 = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.


Please fill in the infomation in the given template below.
Please give answer base on the following API misuse rules.
Please including the following points in your code review comment:
    1. Is the commit a fix of API misuse, if no, please provide the reason
    2. If it is API misuse, what is the action of the fix, if no, please say NA
    3. If it is API misuse, what is the API-element of the fix, if no, please say NA
    4. If it is API misuse, what is the motivation of the fix, if no, please say NA


How to determinte if the commit is a fix of API misuse:
    1. An API misuse is a subset of bug fix.
    2. Renaming and feature addition are not API misuse.
    2. Documents fix and update are not API misuse. 
    3. Any testing related fix are not API misuse.
    4. An API misuse fix has to be code change on API-emelemnts, such as API call, API parameter, and API condition check.
    5. The motivation of the fix has to be in the one of the following categories:
        1. Math fix: fix math error such as devide by zero.
        2. Resource fix: fix resource error such Cuda error, device problem, and CPU and GPU problem.
        3. Shape fix: fix shape error such as Tensor shape mismatch.
        4. State fix: fix state error such as state add torch.no_grad before tensor operation.
        5. Type fix: fix type error such as type mismatch.
        6. Null fix: fix null error such as checking if parameter is null before pass it to API.
        7. Argument fix: fix argument error such as argument missing.
        8. Refactor fix: fix refactor error such as update API call after refactor.
        9. Version fix: fix version error such as update API call after version update.

code change: 
{change_code}

removed code: 
{removed_code}

added code: 
{added_code}

code change explaination:
{code_change_explaination}

template:
if_API_fix: yes or no
action_of_fix: removal, addition, change, or update
API_element_of_fix: API call, API parameter, or API condition check
motivation_of_fix: math, resource, shape, state, type, null, argument, refactor, or version
reason_of_fix: reason of the fix

"""


for i in range(294, len(data)):
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

    # print("commit_message", len(commit_message))
    print("change", len(change))
    print("removed", len(removed))
    print("added", len(added))

    prompt_1 = template_1.format(
        change_code=change, removed_code=removed, added_code=added)
    print("prompt_1", len(prompt_1))
    code_change_explaination = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_1,
        max_tokens=200,
        temperature=0
    )
    code_change_explaination = code_change_explaination["choices"][0]["text"]
    print("code_change_explaination", len(code_change_explaination))
    prompt_2 = template_2.format(change_code=change,
                                 removed_code=removed, added_code=added, code_change_explaination=code_change_explaination)
    print("prompt_2", len(prompt_2))

    output_2 = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_2,
        max_tokens=200,
        temperature=0
    )
    output_2 = output_2["choices"][0]["text"]

    output_2_dict = {}
    try:
        for line in output_2.splitlines():
            if line != "":
                key, value = line.split(":")
                output_2_dict[key] = value
    except:
        pass

    output = {
        "number": number,
        "code_change_explaination": code_change_explaination,
        "misuse_classification": output_2_dict
    }
    print(output)
    with open('C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_classification_v2.json', 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
