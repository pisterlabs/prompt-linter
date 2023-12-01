from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from dotenv import load_dotenv
import os
import json
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_detection_result(detection_result):

    detection_result = detection_result.lower()
    result=""
    if "yes" in detection_result:
        result = "yes"
    elif "no" in detection_result:
        result = "no"
    else: 
        print("error: ", detection_result)

    return result

input_path = "C:\@code\APIMISUSE\data\detection_result_v4_existing.json"
output_path = "C:\@code\APIMISUSE\data\detection_result_yes_v4_existing.json"
# read the file
with open(input_path, encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    data = [json.loads(line) for line in data]

res_list = []
for idx, line in enumerate(data):
    result = line["detection_result"]
    # print(type(result))
    # print("=====================================")
    # print(result)   
    result = parse_detection_result(result) 
    data[idx]["result"] = result
    data[idx]["label"] = ""
    if type(line["code_before"])==str:
        line["code_before"] = line["code_before"].split("\n")
    line["example"] = line["example"].split("\n")
    # line["detection_result"] = line["detection_result"].split("\n")
    res_list.append(line)

result_list = [x["result"] for x in res_list ]
counter = Counter(result_list)
print(counter)
yes_list = [x for x in res_list if x["result"] == "yes"]
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(yes_list, f, indent=4)