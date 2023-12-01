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


# calib
# manual
# test_1
# test_2
# extra
base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\manual\\"
base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\extra\\"
# base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\test_2\\"
input_path = base_path + "manual_data_1k.json"
input_path = base_path + "extra_data_223.json"
# input_path = base_path + "test_2_data_1k.json"
stage_2_path = base_path + "misuse_v3_classification_stage_2.json"
stage_1_path = base_path + "misuse_v2_stage_1_code_explain.json"
output_path = base_path + "misuse_v3_classification_stage_3.json"

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

for key in data_dict.keys():
    if key in data_stage_1_dict.keys():
        if key in data_stage_2_dict.keys():
            data_dict[key]["code_change_explaination"] = data_stage_1_dict[key]["code_change_explaination"]


data = []
for key in data_dict.keys():
    if "code_change_explaination" in data_dict[key].keys():
        data.append(data_dict[key])

# only keep these keys in data: number, change, commit_message, code_change_explaination


for i in range(152, len(data)):
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
You are great at explain code changes in a concise and easy to understand manner.
When you don't know the answer to a question you admit that you don't know.


Task: 
First please read the context, code change explaination, removed code, and added code.
context: 
{change_code}

code change explaination:
{code_change_explaination}

removed code: 
{removed_code}

added code: 
{added_code}    

Then, Please answer the following question and return the answer in given format at the end:

Questions:    
what is the code commit Symptom? please choose one of the options in the ().
    (Program Crash) : The program would crash. 
    (Unexpected Output) : The program output is different from context expectation.
    (Return Warning) : The program returns a warning .
    (Low Efficiency) : The code before change is causing slow execution speed or low performance.
what is the code commit Motivation? please choose one of the options in the <>.
    <Deprecation Management Error> : The code before change is using deprecated API.
    <Data Conversion Error> : The code before change has shape or type missmatch, missing type or dtype specification, or missing type or shape conversion.
    <Device Management Error> : The code before change has device, GPU, CPU, parallelization, or distributed computing related issues.
    <State Handling Error> : The code before change has incorrect state handling such as missing no_grad(), is_training, or eval() API calls
    <Algorithm Error>: The code before change has mathmatical or algorithmic errors such as missing epsilon or atol, incorrect loss function, or incorrect gradient calculation.
    <Null Reference Error> : The code before change missing null check before API call.
what is the action of the fix?
    <Add API> : The fix is adding a new API.
    <Remove API> : The fix is removing an API.
    <Update API> : The fix is updating an API class.
    <change API> : The fix is replace an API with another API.
what is the API-element of the fix?
    <API call>: The fix is change/update/add/remove the API class or method.
    <API parameter>: The fix is change/update/add/remove the API parameteror argument.
    <API condition check>: The fix is change/update/add/remove the if condition check before API method.

    
<answer start>
Symptom:  (Program Crash, Unexpected Output, Return Warning, Low Efficiency)
Motivation: <Deprecation Management Error, Data Conversion Error, Device Management Error, State Handling Error, Algorithm Error, Null Reference Error, Argument error>
Action: <removal, addition, change, or update>
Element: <API call, API parameter, or API condition check>
"""

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

    # stage_3_answer = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "user", "content": prompt_1}
    #     ]
    # )

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
