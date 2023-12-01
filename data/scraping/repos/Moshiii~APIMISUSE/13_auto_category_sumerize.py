import os
import json
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# read
data_dict = {}
with open("C:\@code\APIMISUSE\data\misuse_jsons\manual\merged_split_hunk_AST_filter_manual_deduplica_reduced_category_strict_general_case_Moshi.json", encoding="utf-8") as f:
    data_manual = json.load(f)
    for line in data_manual:
        data_dict[line["number"]] = line

data_stage_2_dict = {}
with open('C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_v2_auto_category.json', encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    data = [json.loads(line) for line in data]
    print(len(data))
    for line in data:
        data_stage_2_dict[line["number"]] = line

# reset file 'C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_v2_compare.json'
with open('C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_v2_compare_auto_category.json', 'w', encoding="utf-8") as f:
    f.write("")


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
print(data[0].keys())

API_method_list = []
with open('C:\@code\APIMISUSE\data\API_method_list_torch.txt', encoding="utf-8") as f:
    API_data = f.readlines()
    for x in API_data:
        if x != "\n":
            API_method_list.append(x.strip().lower())
with open('C:\@code\APIMISUSE\data\API_method_list_tf.txt', encoding="utf-8") as f:
    API_data = f.readlines()
    for x in API_data:
        if x != "\n":
            API_method_list.append(x.strip().lower())

failed_counter = 0
Problem_list = []
print(len(data))

for idx in range(len(data)):
    print(idx)
    if "code_change_explaination" not in data[idx].keys():
        break

    number = data[idx]["number"]
    code_change_explaination = data[idx]["code_change_explaination"]
    res_list = code_change_explaination.split("\n")
    Problem = ""
    Root_cause = ""
    Solution = ""
    for x in res_list:
        if x.startswith("Problem"):
            Problem = x
        if x.startswith("Root_cause") or x.startswith("Root cause") or x.startswith("Root Cause") or x.startswith("Root-Cause:"):
            Root_cause = x
        if x.startswith("Solution"):
            Solution = x
    if Problem == "" or Root_cause == "" or Solution == "":
        failed_counter += 1
    else:
        # print("number: ", number)
        Solution=Solution.replace("Solution: ", "")
        response = openai.Embedding.create(
            input=Problem,
            model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']
        output={
            "embeddings": embeddings,
            "Solution": Solution
        }
        Problem_list.append(output)

with open('C:\@code\APIMISUSE\data\Solution_list.json', 'w', encoding="utf-8") as f:
    json.dump(Problem_list, f, indent=4)
    
