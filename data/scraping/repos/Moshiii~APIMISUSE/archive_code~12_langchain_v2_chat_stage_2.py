
from dotenv import load_dotenv
import os
import json
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# read
data_dict = {}
with open("C:\@code\APIMISUSE\data\misuse_jsons\manual\merged_split_hunk_AST_filter_manual_deduplica_reduced_category.json", encoding="utf-8") as f:
    data_manual = json.load(f)
    for line in data_manual:
        data_dict[line["number"]] = line

data_stage_1_dict = {}
with open('C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_v2_stage_1_code_explain.json', encoding="utf-8") as f:
    data_stage_1 = f.readlines()
    data_stage_1 = [line for line in data_stage_1 if line != "\n"]
    data_stage_1 = [json.loads(line) for line in data_stage_1]
    for line in data_stage_1:
        data_stage_1_dict[line["number"]] = line

# merge data_stage_1_dict into data_dict by key
for key in data_dict.keys():
    if key in data_stage_1_dict.keys():
        data_dict[key]["code_change_explaination"] = data_stage_1_dict[key]["code_change_explaination"]

# convert data_dict to list
data = []
for key in data_dict.keys():
    data.append(data_dict[key])


template_2_1 = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Please including the following points in your code review comment:
    1. Is the changes a fix of API misuse, if no, please say no.
    2. Please explain your reason of the classification judgement.
    please limit your explaination to 3 sentences.

Please give answer base on the following API misuse rules.
How to classify API misuse fix:
    API misuse is rare, please be careful when you classify a fix as API misuse.
    An API misuse fix has to be code change on API-emelemnts, such as API call, API parameter, and API condition check.
    An the reason of API misuse are due to failing to follow the API usage rules and API constraints. 
    Any change in assertion is not API misuse.
    Simple renaming of any elements such as method defination, functions,variable,values, are not API misuse.
    Simple replacing of any elements such as method defination, functions,variable,values for logic fix, are not API misuse.
    Documents changes, logging, printing, and string change are not API misuse. 
    Any test related change are not API misuse, such as key word "test" in def or doc.
    changes in comments are not API misuse.
    If the added and removed only contain return line, method definition (def) or user defined API or class, it is not API misuse.
    If the added and removed do not have API method, it is not API misuse.
    
The following is the code change and similar example for your reference.
please limit your explaination to 3 sentences.
code change: 
{change_code}

code_change_explaination:
{code_change_explaination}

Please fill in the infomation in the given template below.

template:
if_API_misuse_fix: yes or no
reason_of_judgement: reason of the judgement
"""


template_2_2 = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Please including the following points in your code review comment:
    1. Is the changes a fix of API misuse, if no, please say no.
    2. Please explain your reason of the classification judgement.
    please limit your explaination to 3 sentences.

Please give answer base on the following API misuse rules.
How to classify API misuse fix:
    API misuse is rare, please be careful when you classify a fix as API misuse.
    An API misuse fix has to be code change on API-emelemnts, such as API call, API parameter, and API condition check before calling the API such as "with" and "if" statement.
    Any change in assertion or import are not API misuse.
    Simple renaming of any elements such as method defination, functions,variable,values, are not API misuse.
    Simple replacing of any elements such as method defination, functions,variable,values for logic fix, are not API misuse.
    Documents changes, logging, printing, and string change are not API misuse. 
    Any test related change are not API misuse, such as key word "test" in def or doc.
    changes in comments are not API misuse.
    If the added and removed only contain return line, method definition (def) or user defined API or class, it is not API misuse.
    If the added and removed do not have API method, it is not API misuse.
    
The following is the code change and similar example for your reference.
please limit your explaination to 3 sentences.
code change: 
{change_code}

code_change_explaination:
{code_change_explaination}

Please fill in the infomation in the given template below.

template:
if_API_misuse_fix: yes or no
reason_of_judgement: reason of the judgement
"""


template_2_3 = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Please including the following points in your code review comment:
    1. Is the changes a fix of API misuse, if no, please say no.
    2. Please explain your reason of the classification judgement.
    please limit your explaination to 3 sentences.

Please give answer base on the following API misuse rules.
How to classify API misuse fix:
    API misuse fix is rare, please be careful when you classify a fix as API misuse.
    API misuse occurs when the API usage rules and API constraints are not followed.
    API misuse fix are bug fixes that motivated by fixing the violation of API usage rules and API constraints.
    The code before API misuse fix is not working properly, the code after API misuse fix would work.
    An API misuse fix has to be code change on API-emelemnts, such as API call, API parameter, and API condition check before calling the API such as "with" and "if" statement.
    Any change in assertion or import are not API misuse fix. 
    Typo fix are not API misuse fix.
    Simple replacing or renaming of any elements such as method defination, functions, variable, values for logic fix, are not API misuse.
    Documents changes, logging, printing, and string change are not API misuse. 
    Any test related change are not API misuse, such as key word "test" in def or doc.
    changes in comments are not API misuse.
    If the added and removed only contain return line, method definition (def) or user defined API or class, it is not API misuse.
    If the added and removed do not have API method, it is not API misuse.
    
The following is the code change and similar example for your reference.
please limit your explaination to 3 sentences.
code change: 
{change_code}

code_change_explaination:
{code_change_explaination}

Please fill in the infomation in the given template below.

template:
if_API_misuse_fix: yes or no
reason_of_judgement: reason of the judgement
"""


template_2_4 = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Please including the following points in your code review comment:
    1. Is the changes a fix of API misuse, if no, please say no.
    2. Please explain your reason of the classification judgement.
    please limit your explaination to 3 sentences.

Please give answer base on the following API misuse rules.
How to classify API misuse fix:

    Any change in assertion or import are not API misuse fix. 
    Typo fix, addressing scope issue, Feature enhancements are not API misuse fix.
    Documents changes, logging, printing, comments, string and test-related changes are not API misuse. 
    Simple replacing or renaming of any elements such as method defination, functions, variable, values for logic fix, are not API misuse.
    Any test related change are not API misuse, such as key word "test" in def or doc.
    If the added and removed only contain return line, method definition (def) or user defined API or class, it is not API misuse.
    If the added and removed do not have API method, it is not API misuse.
        API misuse fix is rare, please be careful when you classify a fix as API misuse.
    API misuse occurs when the API usage rules and API constraints are not followed.
    API misuse fix are bug fixes that motivated by fixing the violation of API usage rules and API constraints.
    The code before API misuse fix is not working properly, the code after API misuse fix would work.
    
The following is the code change and similar example for your reference.
please limit your explaination to 3 sentences.
code change: 
{change_code}

code_change_explaination:
{code_change_explaination}

Please fill in the infomation in the given template below.

template:
if_API_misuse_fix: yes or no
reason_of_judgement: reason of the judgement
"""

template_2_5 = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Please including the following points in your code review comment:
    1. Is the changes a fix of API misuse, if no, please say no.
    2. Please explain your reason of the classification judgement.
    please limit your explaination to 3 sentences.

Please give answer base on the following API misuse rules.
How to classify API misuse fix:
    Any change in assertion or import are not API misuse fix. 
    Typo fix, addressing scope issue, Feature enhancements are not API misuse fix.
    Documents changes, logging, printing, comments, string and test-related changes are not API misuse. 
    Simple replacing or renaming of any elements such as method defination, functions, variable, values for logic fix, are not API misuse.
    Any test related change are not API misuse, such as key word "test" in def or doc.
    If the added and removed only contain return line, method definition (def) or user defined API or class, it is not API misuse.
    If the added and removed do not have API method, it is not API misuse.
    API misuse fix is rare, please be careful when you classify a fix as API misuse.
    API misuse occurs when the API usage rules and API constraints are not followed.
    API misuse fix are bug fixes that motivated by fixing the violation of API usage rules or API constraints 
    Fixing pre condition of API call by add, remove or change if condition before API call is API misuse fix.
    The code before API misuse fix is not working properly, the code after API misuse fix would work.
    
The following is the code change and similar example for your reference.
please limit your explaination to 3 sentences.
code change: 
{change_code}

code_change_explaination:
{code_change_explaination}

Please fill in the infomation in the given template below.

template:
if_API_misuse_fix: yes or no
reason_of_judgement: reason of the judgement
"""

template_2_6 = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Please including the following points in your code review comment:
    1. Is the changes a fix of API misuse, if no, please say no.
    2. Please explain your reason of the classification judgement.
    please limit your explaination to 3 sentences.

Please give answer base on the following API misuse rules.
How to classify API misuse fix:
    Any change in assertion or import are not API misuse fix. 
    Typo fix, addressing scope issue, Feature enhancements are not API misuse fix.
    Documents changes, logging, printing, comments, string and test-related changes are not API misuse. 
    Simple replacing or renaming of any elements such as method defination, functions, variable, values for logic fix, are not API misuse.
    Any test related change are not API misuse, such as key word "test" in def or doc.
    change method defination is not API misuse fix.
    change return statement is not API misuse fix.
    change user defined API is not API misuse fix.
    if change do not contain API method, it is not API misuse fix.
    API misuse fix is rare, please be careful when you classify a fix as API misuse.
    API misuse occurs when the API usage rules and API constraints are not followed.
    API misuse fix are bug fixes that motivated by fixing the violation of API usage rules or API constraints 
    Fixing if condition before API call is API misuse fix.
    The code before API misuse fix is not working properly, the code after API misuse fix would work.
    
The following is the code change and similar example for your reference.
please limit your explaination to 3 sentences.
code before change: 
{before}

code after change:
{after}

code_change_explaination:
{code_change_explaination}

Please fill in the infomation in the given template below.

template:
if_API_misuse_fix: yes or no
reason_of_judgement: reason of the judgement
"""
# 196
for i in range(0, len(data)):
    item = data[i]
    # for i in range(110, 114):
    print("current_index:", i, "/", len(data))

    # commit_message = "{}\n".format(data[i]["commit_message"])
    change = ""
    removed = ""
    added = ""
    before = ""
    after = ""

    for j in range(0, len(item["change"])):
        line = item["change"][j]
        before += "{}\n".format(item["change"][j])
        after += "{}\n".format(item["change"][j])
        if line.startswith("-"):
            before += "{}\n".format(item["change"][j])
            removed += "{}\n".format(item["change"][j])
        elif line.startswith("+"):
            after += "{}\n".format(item["change"][j])
            added += "{}\n".format(item["change"][j])

    number = item["number"]

    # print("commit_message", len(commit_message))
    print("change", len(change))
    print("removed", len(removed))
    print("added", len(added))
    print("code_change_explaination", len(item["code_change_explaination"]))

    # prompt_2 = template_2_5.format(change_code=change,
    #                                removed_code=removed, added_code=added, code_change_explaination=item["code_change_explaination"])
    prompt_2 = template_2_6.format(
        after=after, before=before, code_change_explaination=item["code_change_explaination"])
    print("prompt_2", len(prompt_2))

    output_2 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt_2}
        ]
    )

    output_2 = output_2["choices"][0]["message"]["content"]

    # output_2 = output_2["choices"][0]["text"]

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
        "code_change_explaination": item["code_change_explaination"],
        "misuse_classification": output_2_dict
    }
    print(output)
    with open('C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_v2_stage_2_classification_template_2_6.json', 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
