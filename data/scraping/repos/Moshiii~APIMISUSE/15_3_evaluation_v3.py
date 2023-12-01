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

    detection_result = detection_result.split("\n")[-1].lower()
    if "yes" in detection_result:
        detection_result = "yes"
    elif "no" in detection_result:
        detection_result = "no"
    else: 
        detection_result = "no"
    return detection_result

# calib
# manual
base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\manual\\"
# input_path = base_path + "calib_data_1k.json"
# input_path = base_path + "misuse_v3_classification_stage_3_result.json"

# predict_path = base_path + "detection_result_v3.json"
predict_path = base_path + "detection_result.json"
example_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\manual\misuse_report.json"
# label_path = base_path + "calib_invest_data_1k.json"
label_path = base_path + "manual_invest_data_1k.json"
debug_output_path = base_path + "debug_false_positive.json"

# data_Root_Cause = []
# data_dict={}
# with open(input_path, encoding="utf-8") as f:
#     data = json.load(f)
#     for line in data:
#         data_Root_Cause.append(line["Root_Cause"])
#         data_dict[line["number"]] = line
# print("data_Root_Cause: ", len(data_Root_Cause))

example_dict = {}
with open(example_path, encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    data = [json.loads(line) for line in data]
    for line in data:
        item = {
            "number": line["number"],
            "change": line["change"],
            "report": line["report"],
        }
        example_dict[line["number"]] = item


data_label = []
with open(label_path, encoding="utf-8") as f:
    data = json.load(f)
    for line in data:
        item = {
            "number": line["number"],
            "label": line["label"],
            "change": line["change"],
        }
        data_label.append(item)
print("data_label: ", len(data_label))

data_predict = {}
with open(predict_path, encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    data = [json.loads(line) for line in data]
    for line in data:
        data_predict[line["number"]] = line

data = []
for key in data_predict.keys():
    data.append(data_predict[key])

# only keep these keys in data: number, change, commit_message, code_change_explaination
for i in range(0, len(data)):
    detection_reason = data[i]["detection_result"]
    detection_result = parse_detection_result(detection_reason)
    data[i] = {
        "number": data[i]["number"],
        "detection_result": detection_result,
        "detection_reason": detection_reason,
        "prompt_2": data[i]["prompt_2"],
    }


predict_list_all = data[:185]
label_list_all = data_label[: len(predict_list_all)]

result_list = []
for idx in range(0, len(predict_list_all)):
    label = label_list_all[idx]["label"]
    predict = predict_list_all[idx]["detection_result"]
    detection_reason = predict_list_all[idx]["detection_reason"]
    item = {
        "number": label_list_all[idx]["number"],
        "label": label,
        "predict": predict,
        "report": example_dict[label_list_all[idx]["number"]]["report"].split("\n"),
        "decision": detection_reason.split("\n"),
        "prompt_2": predict_list_all[idx]["prompt_2"].split("\n"),
    }
    if label == "no" and label != predict:
        result_list.append(item)

with open(debug_output_path, "w", encoding="utf-8") as f:
    json.dump(result_list, f, indent=4, ensure_ascii=False)


predict_list = [x["detection_result"] for x in predict_list_all]
label_list = [x["label"] for x in label_list_all]

print(Counter(label_list))
print(Counter(predict_list))

print(confusion_matrix(label_list, predict_list))
print(accuracy_score(label_list, predict_list))


# print(Counter(data_Root_Cause))
# class_name_list = ['Deprecation Management Error', 'Data Conversion Error',
#                    'Algorithm Error', 'Null Reference Error', 'Argument Error', 'State Handling Error']
# for class_name in class_name_list:
#     print("class_name: ", class_name)
#     mask = [x == class_name for x in data_Root_Cause]

#     predict_list = [predict_list_all[i]
#                     for i in range(0, len(predict_list_all)) if mask[i]]
#     label_list = [label_list_all[i]
#                   for i in range(0, len(label_list_all)) if mask[i]]

#     predict_list = [x["detection_result"] for x in predict_list]
#     label_list = [x["label"] for x in label_list]

#     if len(label_list) == 0:
#         continue
#     if len(predict_list) == 0:
#         continue
#     print(Counter(label_list))
#     print(Counter(predict_list))
#     print(confusion_matrix(label_list, predict_list))
#     print(accuracy_score(label_list, predict_list))
