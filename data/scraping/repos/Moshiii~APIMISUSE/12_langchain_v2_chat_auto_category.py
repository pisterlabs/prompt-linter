from dotenv import load_dotenv
import os
import json
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# read
data_dict = {}
with open("C:\@code\APIMISUSE\data\misuse_jsons\manual\merged_split_hunk_AST_filter_manual_deduplica_reduced_category_strict_general_case_Moshi.json", encoding="utf-8") as f:
    data_manual = json.load(f)
    for line in data_manual:
        data_dict[line["number"]] = line

data_stage_2_dict = {}
with open('C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_v2_stage_1_code_explain.json', encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    data = [json.loads(line) for line in data]
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

data=[x for x in data if x["label"]=="yes"]
print(len(data))
template_1 = """
You are an experienced software developer.
You are great at explain code changes in a concise and easy to understand manner.
When you don't know the answer to a question you admit that you don't know.

Please read the following code commit and provide a code change explaination.
The context section is the context of the code change.
The removed lines of code start with "-" and the added lines of code start with "+".

Please answer the following question considering the context, code change explaination, removed code, and added code and return the answer in given format:
    
Questions:    
what is the code commit Symptom? please choose one of the options in the ()
    (Program Crash) : The program crashes when the code before change is executed. 
    (Incorrect Output) : The program output is incorrect when the code before change is executed.
    (Return Warning) : The program returns a warning when the code before change is executed.
    (Low Efficiency) : The code before change is inefficient.
what is the code commit Motivation? please choose one of the options in the <>
    <Deprecation Management Error> : The code before change is using deprecated API.
    <Data Conversion Error> : The code before change has shape or type missmatch, missing type or dtype specification, or missing type or shape conversion.
    <Device Management Error> : The code before change has device, GPU, CPU, parallelization, or distributed computing related issues.
    <State Handling Error> : The code before change has incorrect state handling such as missing no_grad(), is_training, or eval() API calls
    <Algorithm Error>: The code before change has mathmatical or algorithmic errors such as missing epsilon or atol, incorrect loss function, or incorrect gradient calculation.
    <Null Reference Error> : The code before change missing null check before API call.


context: 
{change_code}

code change explaination:
{code_change_explaination}

removed code: 
{removed_code}

added code: 
{added_code}

<answer start>
Symptom:  (Program Crash, Incorrect Output, Return Warning, Low Efficiency)
Motivation: <Deprecation Management Error, Data Conversion Error, Device Management Error, State Handling Error, Algorithm Error, Null Reference Error, Argument error>
"""

#1032
for i in range(69, len(data)):
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
        change_code=change, removed_code=removed, added_code=added,code_change_explaination=code_change_explaination)
    print("prompt_1", len(prompt_1))

    
    code_change_explaination = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "user", "content": prompt_1}
        ]
    )
    code_change_explaination = code_change_explaination["choices"][0]["message"]["content"]

    output = {
        "number": number,
        "code_change_explaination": code_change_explaination,
    }

    with open('C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_v2_auto_category_2.json', 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
