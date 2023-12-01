import re
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


client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                  persist_directory="C:\@code\APIMISUSE\chroma_db"
                                  ))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-ada-002"
)
# database_name = "API_misuse_fix_pattern_rules"
database_name = "fix_rules_pattern_enbedding_900"
collection = client.get_or_create_collection(
    database_name, embedding_function=openai_ef)
collection_list = client.list_collections()
print(collection_list)


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def get_similar_example_id(code_before, top_k=4):
    results = collection.query(
        query_texts=code_before,
        n_results=top_k,
    )
    return results["ids"]


input_path = "C:\@code\APIMISUSE\data\API_call_10_latest.json"
example_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\data_all\\fix_rules.json"
output_path = "C:\@code\APIMISUSE\data\detection_result_v3.json"


def parse_fix_pattern(input_str):
    input_str = input_str.replace("\n", "").lower().replace(
        "fix pattern:", "fix_pattern:")
    if "fix_pattern:" in input_str:
        output_str = input_str.split("fix_pattern:")[1]
    else:
        print("error: fix_pattern: not found")
        output_str = ""
    return output_str


def get_APIs(change):
    API_list = []
    change = change.split("\n")
    change = [x for x in change if not x.startswith("+")]
    change = [x[1:] for x in change if not x.startswith("-")]
    change = "\n".join(change)
    # print(change)
    API = re.findall(r"\.[a-zA-Z0-9_]+\(", change)
    if len(API) > 0:
        API = API[0]
        API_list.append(API)
    return API_list

# read
example_dict = {}
with open(example_path, encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    data = [json.loads(line) for line in data]
    for line in data:
        pattern = parse_fix_pattern(line["fix_pattern"])
        APIs = get_APIs(line["change"])
        # print(APIs)
        item = {
            "number": line["number"],
            "change": line["change"],
            "fix_pattern": pattern,
            "APIs": APIs,
        }
        # print(pattern)
        example_dict[line["number"]] = item
print(len(example_dict.keys()))
data_dict = {}
with open(input_path, encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    for x in range(0, len(data)):
        data[x] = json.loads(data[x])
    for line in data:
        item = {
            "file_path": line["file_path"],
            "line_number": line["line_number"],
            "number": line["file_path"]+":"+str(line["line_number"]),
            "API": line["API"],
            "context": line["context"],
        }
        data_dict[item["number"]] = item

data = []
for key in data_dict.keys():
    data.append(data_dict[key])

print(len(data))
print(data[0].keys())
# exit()


def parse_fix_pattern(input_str):
    input_str = input_str.replace("\n", "").lower().replace(
        "fix pattern:", "fix_pattern:")
    if "fix_pattern:" in input_str:
        output_str = input_str.split("fix_pattern:")[1]
    else:
        print("error: fix_pattern: not found")
        output_str = ""
    return output_str


template_1 = """
As an experienced software developer, you have a strong ability to read and understand code snippet. If you encounter a question to which you don't know the answer, you acknowledge your lack of knowledge.

Task:
Descript what the following code snippet does in two sentence.

Code snippet:
{code_before}

Answer:
"""

template_2 = """

Please read the following code snippet and fix rules. Then, answer if the fix pattern can be applied in the code snippet.
If pattern can be applied, answer "Yes",  if not, answer "No" directly.

Code snippet:
{code_before}

Fix rules:
{fix_rules}

Decision:(Yes/No)
"""
# 14
# for i in range(0, len(data)):
for i in range(0, 2):
    print("current_index:", i, "/", len(data))
    code_before = ""
    for x in data[i]["context"]:
        code_before += x 
    number = data[i]["number"]
    print("number", number)
    # print("code_before", len(code_before))
    if len(code_before) == 0:
        # reasoning == "NA"
        example_prompt = "NA"
        prompt_2 = "NA"
    else:
        query_API = data[i]["API"]
        example_dict_filtered = {}
        for x in example_dict.items():
            API_example_list = x[1]["APIs"]
            number_x = x[1]["number"]
            if query_API in API_example_list:
                example_dict_filtered[number_x] = x[1]
        # print(len(example_dict_filtered))
        fix_pattern_prompt = ""
        idx = 0
        for x in example_dict_filtered.items():
            if idx>=4:
                continue
            idx+=1
            fix_pattern = x[1]["fix_pattern"]
            fix_pattern_prompt +="Fix pattern "+ str(idx) +" :\n"
            fix_pattern_prompt += fix_pattern + "\n"
        # print("fix_pattern_prompt", fix_pattern_prompt)

        # example_list = get_similar_example_id(answer, top_k=1)
        # id = int(example_list[0][0])
        # example = example_dict[id]
        # example_prompt = parse_fix_pattern(example["fix_pattern"].lower())

        prompt_2 = template_2.format(
            code_before=code_before, fix_rules=fix_pattern_prompt)
        prompt_2 = prompt_2.replace("\n\n", "\n")
        # print("prompt_2", len(prompt_2))
        print("prompt_2", prompt_2)

        # decition = openai.ChatCompletion.create(
        decition = completion_with_backoff(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt_2}
            ]
        )
        decition = decition["choices"][0]["message"]["content"]

    output = {
        "number": number,
        "code_before": code_before,
        "example": fix_pattern_prompt,
        "detection_result": decition,
        "prompt_2": prompt_2,
    }

    with open(output_path, 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
