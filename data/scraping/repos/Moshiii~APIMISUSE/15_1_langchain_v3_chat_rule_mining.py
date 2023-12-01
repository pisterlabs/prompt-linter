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
output_path = base_path + "misuse_fix_pattern_rules.json"

# read
data_dict = {}
with open(input_path, encoding="utf-8") as f:
    data_manual = json.load(f)
    for line in data_manual:
        data_dict[line["number"]] = line

data = []
for key in data_dict.keys():
        if data_dict[key]["label"] == "yes":
            data.append(data_dict[key])

# only keep these keys in data: number, change, commit_message, code_change_explaination
for i in range(0, len(data)):
    data[i] = {
        "change": data[i]["change"],
        "number": data[i]["number"],
    }


print(len(data))
print(data[0].keys())


template_1 = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Task: 
First please read the given code change example below and the fix pattern after it.


code change example:
\"\"\"
class TestMotionBlur:
) -> torch.Tensor:
return kornia.filters.motion_blur(input, ksize, angle, direction)
-        img = torch.rand(2, 3, 4, 5)
+        img = torch.rand(2, 3, 4, 5).to(device)
ksize = 5
angle = 65.
direction = .1
\"\"\"

fix_pattern example:
\"\"\"
if  torch.rand() dtected, replace it with torch.rand().to(device)
\"\"\"

Now, please read the code change example below and generate the fix pattern.
code change: 
{change_code}

fix_pattern: 
"""
# 48
for i in range(0, len(data)):
    print("current_index:", i, "/", len(data))
    change = ""

    for j in range(0, len(data[i]["change"])):
        line = data[i]["change"][j]
        change += "{}\n".format(data[i]["change"][j])
    number = data[i]["number"]
    print("number", number)
    print("change", len(change))

    prompt_1 = template_1.format(
        change_code=change)
    print("prompt_1", len(prompt_1))

    fix_rule = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt_1}
        ]
    )

    fix_rule = fix_rule["choices"][0]["message"]["content"]

    output = {
        "number": number,
        "change": change,
        "fix_rule": fix_rule,
    }

    with open(output_path, 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
