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
database_name = "fix_rules"
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
# input_path = base_path + "manual_data_1k.json"
example_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\manual\\fix_rules.json"
# input_path = base_path + "misuse_v3_classification_stage_3_result.json"
output_path = "C:\@code\APIMISUSE\data\detection_result.json"

# read
example_dict = {}
with open(example_path, encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    data = [json.loads(line) for line in data]
    for line in data:
        item = {
            "number": line["number"],
            "change": line["change"],
            "fix_pattern": line["fix_pattern"],
        }
        example_dict[line["number"]] = item

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

template_2 = """
As an experienced software developer, you have a strong ability to read and understand code snippet. If you encounter a question to which you don't know the answer, you acknowledge your lack of knowledge.

Task:
First, carefully review the provided code snippet and its explanation. Then, provide your answer as to whether the code exhibits API misuse or not.

Based on the information provided, please read the following code snippet and the fixing rule. 
Think carefully step by step and answer whether the fixing rule applys to the given code snippet.

First, check if the condition of the fixing rule can be identified in the code snippet. if not, answer "No" directly.
Then, check if the Pattern in the fixing rule can be identified code snippet. if not, answer "No" directly.
if the condition and the pattern can be identified in the code snippet, answer "Yes" and provide your reasoning.
Code snippet:
{code_before}

Fix rules:
{fix_rules}


Please answer with the following template:

Reasoning:(please be concise)
Decision:(Yes/No)
"""
# 1358
for i in range(3576, len(data)):
    print("current_index:", i, "/", len(data))
    code_before="" 
    for x in data[i]["context"]:
        code_before += x + "\n"
    number = data[i]["number"]
    print("number", number)
    print("code_before", len(code_before))
    if len(code_before) ==0:
        reasoning=="NA"
        example_prompt="NA"
        prompt_2="NA"

    else:    
        
        example_list = get_similar_example_id(code_before, top_k=1)
        id = int(example_list[0][0])
        example = example_dict[id]
        example_prompt = example["fix_pattern"].lower()
        example_prompt_list = example_prompt.split("\n")
        condition_prompt = ""
        pattern_prompt = ""
        for line in example_prompt_list:
            if line.startswith("<condition") or line.startswith("condition"):
                condition_prompt = line
            if line.startswith("<pattern") or line.startswith("pattern"):
                pattern_prompt = line
        if condition_prompt == "" or pattern_prompt == "":
            print("error!!!!!!!!!!!!!!!!!!!!!!!!!", condition_prompt, pattern_prompt)
        example_prompt_cut = condition_prompt + "\n" + pattern_prompt
        prompt_2 = template_2.format(
            code_before=code_before, fix_rules=example_prompt)
        print("prompt_2", len(prompt_2))
        reasoning = openai.ChatCompletion.create(
        # reasoning = completion_with_backoff(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt_2}
            ]
        )
        reasoning = reasoning["choices"][0]["message"]["content"]

    output = {
        "number": number,
        "code_before": code_before,
        "example": example_prompt,
        "detection_result": reasoning,
        "prompt_2": prompt_2,
    }

    with open(output_path, 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
