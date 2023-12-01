from collections import Counter 
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
# example_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\data_all\\fix_rules_v4_list.json"
output_path = "C:\@code\APIMISUSE\data\detection_result_v4_existing.json"



example_dict = {
    "1": {"APIs": [".decode_jpeg("], "fix_pattern": "in the condition of using a png file, if .decode_jpeg() detected, replace .decode_jpeg() with .decode_image()"},
    "2": {"APIs": [".resize("], "fix_pattern": "in the condition of using a png file, if .decode_jpeg() and .resize( detected, replace .decode_jpeg() with .decode_image() and replace .resize() with .resize_image_with_crop_or_pad("},
    "3": {"APIs": [".merge_summary("], "fix_pattern": "no precondition required, if tf.merge_summary( detected, replace it with tf.summary.merge("},
    "4": {"APIs": [".merge_all_summaries("], "fix_pattern": "no precondition required, if tf.merge_all_summaries( detected, replace it with tf.summary.merge_all("},
    "5": {"APIs": [".SummaryWriter("], "fix_pattern": "no precondition required, if tf.train.SummaryWriter(detected, replace it with tf.summary.FileWriter("},
    "6": {"APIs": [".Dense("], "fix_pattern": "if the model is using binary class mode, if Dense(3, activation=`softmax`) detected, replace it with Dense(2, activation=`softmax`)"},
    "7": {"APIs": [".softmax("], "fix_pattern": "in the condition of using cross entropy formula, if .softmax() detected, replace it tf.nn.softmax_cross_entropy_with_logits()"},
    "8": {"APIs": ["@tf.function"], "fix_pattern": "in the condition of using @tf.function, add tf.range( and tf.cast() to API parameters"},
    "9": {"APIs": [".histogram_summary("], "fix_pattern": "no precondition required, if tf.histogram_summary( is detected, replace it with tf.summary.histogram("},
    "10": {"APIs": [".scalar_summary("], "fix_pattern": "no precondition required, if tf.scalar_summary( is detected, replace it with tf.summary.scalar("},
    "11": {"APIs": [".start_queue_runners("], "fix_pattern": "in the condition of using eval() before calling tf.train.start_queue_runners(x), if tf.start_queue_runners( is detected, Insert a call to tf.train start_queue_runners(x) before eval() and after the most recent assignment of variable x"},
}
example_list = []
for x in example_dict.values():
    example_list.append(x)

data_dict = {}
with open(input_path, encoding="utf-8") as f:
    data = json.loads(f.read())
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
for i in range(1193, len(data)):
    # for i in range(0, 2):
    print("current_index:", i, "/", len(data))
    code_before = ""
    for x in data[i]["context"]:
        code_before += x
    number = data[i]["number"]
    print("number", number)
    if code_before == "":
        continue

    query_API = data[i]["API"]
    print("query_API", query_API)
    example_list_filtered = example_list
    print(len(example_list_filtered))
    fix_pattern_prompt = ""
    idx = 0

    for x in example_list_filtered:

        idx += 1
        fix_pattern = x["fix_pattern"]
        fix_pattern_prompt += "Fix pattern " + str(idx) + " :\n"
        fix_pattern_prompt += fix_pattern + "\n"
    prompt_2 = template_2.format(
        code_before=code_before, fix_rules=fix_pattern_prompt)
    prompt_2 = prompt_2.replace("\n\n", "\n")
    print("prompt_2", len(prompt_2))

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
        # "prompt_2": prompt_2,
    }

    with open(output_path, 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)