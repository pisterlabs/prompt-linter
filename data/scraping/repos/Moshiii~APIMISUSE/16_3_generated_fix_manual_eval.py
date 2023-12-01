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
    result = ""
    if "decision: yes" in detection_result:
        result = "yes"
    elif "decision: no" in detection_result:
        result = "no"
    else:
        print("error: ", detection_result)

    return result


input_path = "C:\@code\APIMISUSE\data\generate_fix_yes_v4_existing.json"
output_path = "C:\@code\APIMISUSE\data\generate_fix_yes_v4_manual_existing.json"
# read the file
with open(input_path, encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    data = [json.loads(line) for line in data]

print("data length: ", len(data))
res_list = []
for idx, line in enumerate(data):
    data[idx]["label"] = ""
    # data[idx]["example"] = data[idx]["example"].split("\n")
    # data[idx]["prompt_2"] = data[idx]["prompt_2"].split("\n")
    data[idx]["result"] = parse_detection_result(data[idx]["Fixed"])
    data[idx]["Fixed"] = data[idx]["Fixed"].split("\n")
    if data[idx]["result"] == "yes" and data[idx]["example"]!=[""]:
        line["example"] = "omit"
        res_list.append(line)
print("data length: ", len(res_list))


with open(output_path, "w", encoding="utf-8") as f:
    json.dump(res_list, f, indent=4)
