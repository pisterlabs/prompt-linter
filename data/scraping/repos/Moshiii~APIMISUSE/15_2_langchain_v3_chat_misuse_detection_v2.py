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

collection = client.get_or_create_collection(
    "API_misuse_fix_pattern_rules", embedding_function=openai_ef)
collection_list = client.list_collections()
print(collection_list)


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def get_similar_example(code_before, top_k=4):
    results = collection.query(
        query_texts=code_before,
        n_results=top_k,
    )
    return results["documents"]


base_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\calib\\"
input_path = base_path + "calib_data_1k.json"
# input_path = base_path + "misuse_v3_classification_stage_3_result.json"

output_path = base_path + "detection_result.json"

# read
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


template_1 = """
As an experienced software developer, you have a strong ability to read and understand code changes. 
If you encounter a question to which you don't know the answer, you acknowledge your lack of knowledge.

Task:
Please carefully review the provided code snippet and provide a concise explanation of its meaning in less than three sentences.

Code snippet:
{code_before}

Code understanding:
"""

template_2 = """
As an experienced software developer, you have a strong ability to read and understand code changes. If you encounter a question to which you don't know the answer, you acknowledge your lack of knowledge.

Positive signs of API misuse:
API misuse refers to using an API in unintended ways, such as incorrect shape type, dtype, or missing null reference checks, mishandling GPU, CPU, parallelization, or distributed computing, incorrect state handling related to no_grad(), is_training, or eval(), missing epsilon or atol, incorrect loss function, or gradient calculation, and incorrect usage of if or with statements just before calling an API.

Task:
First, carefully review the provided code snippet and its explanation. Then, provide your answer as to whether the code exhibits API misuse or not.

Code snippet:
{code_before}

Code understanding:
{code_understanding}

Based on the information provided, please answer whether the given code piece demonstrates API misuse or not.

Answer: (yes/no)
Reason: (Please provide a concise explanation as to why you believe it is API misuse or not in one sentence)
"""


for i in range(0, len(data)):
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
    print("code_before", len(code_before))
    print("code_after", len(code_after))

    prompt_1 = template_1.format(
        code_before=code_before)
    print("prompt_1", len(prompt_1))

    code_understanding = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt_1}
        ]
    )

    code_understanding = code_understanding["choices"][0]["message"]["content"]

    example_list = get_similar_example(code_before, top_k=1)
    print("example_list", example_list)
    examples = ""
    for idx, single_example in enumerate(example_list[0]):
        examples += "'''"+"example_"+str(idx)+"\n"
        single_example = single_example.split("\n")
        for line in single_example:
            examples += "{}\n".format(line)
        examples += "'''"
    print(examples)

    prompt_2 = template_2.format(
        code_before=code_before, code_understanding=code_understanding)
    print("prompt_2", len(prompt_2))

    if_api_misuse = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt_2}
        ]
    )

    if_api_misuse = if_api_misuse["choices"][0]["message"]["content"]

    output = {
        "number": number,
        "code_before": code_before,
        # "code_after": code_after,
        "code_understanding": code_understanding,
        "example": examples,
        "detection_result": if_api_misuse,
    }

    with open(output_path, 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
