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


# calib
# manual
# base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\manual\\"
base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\calib\\"
input_path = base_path + "calib_data_1k.json"
# input_path = base_path + "manual_data_1k.json"
example_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\manual\\fix_rules.json"
# input_path = base_path + "misuse_v3_classification_stage_3_result.json"
output_path = base_path + "detection_result.json"

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
    data_manual = json.load(f)
    for line in data_manual:
        data_dict[line["number"]] = line

data = []
for key in data_dict.keys():
    data.append(data_dict[key])

# only keep these keys in data: number, change, commit_message, code_change_explaination
for i in range(0, len(data)):
    data[i] = {
        "change": data[i]["change"],
        "number": data[i]["number"],
    }


print(len(data))
print(data[0].keys())


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

Reasoning:(please be concise)
Decision:(Yes/No)
"""

template_3 = """
Please read the following paragraph and answer if the paragraph confirms an API misuse or reject an API misuse.

paragraph:
{paragraph}

Answer:(Yes/No)
"""

# 374
for i in range(478, len(data)):
    print("current_index:", i, "/", len(data))
    code_before = ""
    code_after = ""

    for j in range(0, len(data[i]["change"])):
        line = data[i]["change"][j]
        if line.startswith("+"):
            code_after += "{}\n".format(line)
        elif line.startswith("-"):
            code_before += "{}\n".format(line)
        else:
            code_after += "{}\n".format(line)
            code_before += "{}\n".format(line)

    number = data[i]["number"]
    print("number", number)
    print("change", data[i]["change"])
    print("code_before", len(code_before))
    print("code_after", len(code_after))
    if len(code_before) ==0:
        # reasoning=="NA"
        example_prompt="NA"
        prompt_2="NA"

    else:    
        # prompt_1 = template_1.format(
        #     code_before=code_before)
        # print("prompt_1", len(prompt_1))

        # code_understanding = completion_with_backoff(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "user", "content": prompt_1}
        #     ]
        # )

        # code_understanding = code_understanding["choices"][0]["message"]["content"]

        example_list = get_similar_example_id(code_before, top_k=1)
        # print("example_list", example_list)
        id = int(example_list[0][0])
        # print("id", id)
        example = example_dict[id]
        change = example["change"]
        # print("change", change)
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

        # prompt_3 = template_3.format(
        #     paragraph=reasoning)
        # print("prompt_3", len(prompt_3))
        # # reasoning = openai.ChatCompletion.create(
        # answer = completion_with_backoff(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "user", "content": prompt_3}
        #     ]
        # )
        # answer = answer["choices"][0]["message"]["content"]

    output = {
        "number": number,
        "code_before": code_before,
        "code_after": code_after,
        "example": example_prompt,
        "detection_result": reasoning,
        "prompt_2": prompt_2,
        # "answer": answer,
    }

    with open(output_path, 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
