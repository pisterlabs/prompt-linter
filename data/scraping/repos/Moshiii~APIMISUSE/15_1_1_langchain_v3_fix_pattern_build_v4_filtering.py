from dotenv import load_dotenv
import os
import json
import openai
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_APIs(change):
    API_list = []
    API = re.findall(r"\.[a-zA-Z0-9_]+\(", change)
    if len(API) > 0:
        API_list = list(set(API))
    return API_list

input_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\data_all\\fix_rules_v4.json"
output_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\data_all\\fix_rules_v4_list.json"

# read
data_dict = {}
with open(input_path, encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    data = [json.loads(line) for line in data]
    for x in data:
        data_dict[x["number"]] = x  

data = []
for key in data_dict.keys():
    item = data_dict[key]
    item["APIs"] = get_APIs(item["change"])
    if len(item["APIs"])==0:
        continue
    data.append(data_dict[key])
print(len(data))
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)
