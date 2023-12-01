from dotenv import load_dotenv
import os
import json
import openai
load_dotenv()

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(4))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)
# calib
# test_1
# extra
# base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\test_2\\"
base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\\extra\\"
# input_path = base_path + "calib_data_1k.json"
input_path = base_path + "extra_data_223.json"
output_path = base_path + "misuse_v2_stage_1_code_explain.json"
# read
with open(input_path, encoding="utf-8") as f:
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

#1032
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

    # print("commit_message", len(commit_message))
    print("change", len(change))
    print("removed", len(removed))
    print("added", len(added))

    prompt_1 = template_1.format(
        change_code=change, removed_code=removed, added_code=added)
    print("prompt_1", len(prompt_1))


    # code_change_explaination = openai.Completion.create(
    #     model="text-davinci-003",
    #     prompt=prompt_1,
    #     max_tokens=200,
    #     temperature=0
    # )
    
     
    code_change_explaination = completion_with_backoff(
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

    with open(output_path, 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
