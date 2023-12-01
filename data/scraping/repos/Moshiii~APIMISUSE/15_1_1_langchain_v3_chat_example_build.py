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
output_path = base_path + "misuse_report.json"

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


template_yes = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Task: 
please read the API misuse decition example below and generate a report by answering the question step by step to explain the reasoning using the template at the end.

API misuse decition example: 
{change_code}

question:
why the above example is API misuse?
what is the API method involved in the API misuse?
what is the fix pattern for the API misuse? 

TEMPLATE:
Why:
API_method:
Fix_pattern:
"""

template_no = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Task: 
please read the API misuse decition example below and generate a report by answering the question step by step to explain the reasoning.

API misuse decition example: 
{change_code}

question:
why the above example is not API misuse?

Answer:
"""



# 48
for i in range(0, len(data)):
    print("current_index:", i, "/", len(data))
    change = ""

    for j in range(0, len(data[i]["change"])):
        line = data[i]["change"][j]
        change += "{}\n".format(data[i]["change"][j])

    number = data[i]["number"]
    label = data[i]["label"]
    change += " Decision: "
    if label == "yes":
        change += "Yes the given example is an API misuse\n"
    elif label == "no":
        change += "No the given example is not an API misuse\n"
    print("number", number)
    print("change", len(change))

    if label == "yes":
        prompt_1 = template_yes.format(
            change_code=change)
        print("prompt_1", len(prompt_1))

    elif label == "no":
        prompt_1 = template_no.format(
            change_code=change)
        print("prompt_1", len(prompt_1))

    report = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt_1}
        ]
    )

    report = report["choices"][0]["message"]["content"]

    output = {
        "number": number,
        "change": change,
        "report": report,
    }

    with open(output_path, 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
