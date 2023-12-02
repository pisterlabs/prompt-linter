
from dotenv import load_dotenv
import os
import json
import openai
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import chromadb
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_embedding_vectors():
    with open("C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_v2_agree.json", encoding="utf-8") as f:
        data = f.readlines()
        data = [line for line in data if line != "\n"]
        data = [json.loads(line) for line in data]
        # print(data)
    documents = []
    ids = []
    for idx in range(len(data)):
        manual_comments = data[idx]["manual_comments"]
        change = data[idx]["change"]
        number = data[idx]["number"]
        added = data[idx]["added"]
        removed = data[idx]["removed"]
        manual_label = data[idx]["manual_label"]
        doc = ""
        doc += "manual_comments: \n"
        doc += manual_comments
        doc += "change: \n"
        doc += change
        doc += "added: \n"
        doc += added
        doc += "removed: \n"
        doc += removed
        doc += "manual_label: \n"
        doc += manual_label
        documents.append(doc)
        ids.append(str(number))
    # print(documents)
    # print(ids)
    # print(len(documents))
    collection = get_VDB()
    collection.add(
        documents=documents,
        ids=ids
    )


def get_VDB():

    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        # Optional, defaults to .chromadb/ in the current directory
        persist_directory="C:\\@code\\APIMISUSE\\data\\embedding\\data_API_fix_categorizor"

    ))
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ['OPENAI_API_KEY'],
        model_name="text-embedding-ada-002"
    )

    collection = client.get_or_create_collection(
        "langchain", embedding_function=openai_ef)

    return collection

# only run it for one time
# generate_embedding_vectors()


template_1 = """
You are an experienced software developer.
You are great at explain code changes in a concise and easy to understand manner.
When you don't know the answer to a question you admit that you don't know.

Please provide read the following code commit and provide a code change explaination.
Please including the following points in your code change explaination:
    The motivation of the code change.
    The solution to the code change.
please limit your explaination to 3 sentences.

code change: 
{change_code}

removed code: 
{removed_code}

added code: 
{added_code}

code change explaination:
"""
template_3 = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Please including the following points in your code review comment:
    1. Is the changes a fix of API misuse, if no, please say no.
    2. If it is API misuse, what is the action of the fix, if no, please say NA.
    3. If it is API misuse, what is the API-element of the fix, if no, please say NA.
    4. If it is API misuse, what is the motivation of the fix, if no, please say NA.

Please give answer base on the following API misuse rules.
How to classify API misuse fix:
    API misuse is rare, please be careful when you classify a fix as API misuse.
    Simple renaming of any elements such as functions,variable,values, are not API misuse.
    Documents changes and update are not API misuse. 
    Any testing related fix are not API misuse.
    changes in comments are not API misuse.
    If the added and removed only contain return line, method definition (def) or user defined API or class, it is not API misuse.
    If the added and removed do not have API method, it is not API misuse.
    An API misuse fix has to be code change on API-emelemnts, such as API call, API parameter, and API condition check.
    The motivation of the fix has to be in the one of the following categories:
        1. Math fix: fix math error such as devide by zero.
        2. Resource fix: fix resource error such Cuda error, device problem, and CPU and GPU problem.
        3. Shape fix: fix shape error such as Tensor shape mismatch.
        4. State fix: fix state error such as state add torch.no_grad before tensor operation.
        5. Type fix: fix type error such as type mismatch.
        6. Null fix: fix null error such as checking if parameter is null before pass it to API.
        7. Argument fix: fix argument error such as argument missing.
        8. Refactor fix: fix refactor error such as update API call after refactor.
        9. Version fix: fix version error such as update API call after version update.

The following is the code change and similar example for your reference.
please limit your explaination to 3 sentences.
code change: 
{change_code}

removed code: 
{removed_code}

added code: 
{added_code}

code_change_explaination:
{code_change_explaination}

similar example:
{similar_example}

Please fill in the infomation in the given template below.

template:
if_API_misuse_fix: yes or no
action_of_fix: removal, addition, change, or update
API_element_of_fix: API call, API parameter, or API condition check
motivation_of_fix: math, resource, shape, state, type, null, argument, refactor, or version
reason_of_fix: reason of the fix

"""


template_2 = """
You are an experienced software developer.
You are great at read and understand code changes.
When you don't know the answer to a question you admit that you don't know.

Please including the following points in your code review comment:
    1. Is the changes a fix of API misuse, if no, please say no.
    2. Please explain your reason of the classification judgement.
    please limit your explaination to 3 sentences.

Please give answer base on the following API misuse rules.
How to classify API misuse fix:
    API misuse is rare, please be careful when you classify a fix as API misuse.
    An API misuse fix has to be code change on API-emelemnts, such as API call, API parameter, and API condition check.
    An the reason of API misuse are due to failing to follow the API usage rules and API constraints. 
    Any change in assertion is not API misuse.
    Simple renaming of any elements such as method defination, functions,variable,values, are not API misuse.
    Simple replacing of any elements such as method defination, functions,variable,values for logic fix, are not API misuse.
    Documents changes, logging, printing, and string change are not API misuse. 
    Any test related change are not API misuse, such as key word "test" in def or doc.
    changes in comments are not API misuse.
    If the added and removed only contain return line, method definition (def) or user defined API or class, it is not API misuse.
    If the added and removed do not have API method, it is not API misuse.
    
The following is the code change and similar example for your reference.
please limit your explaination to 3 sentences.
code change: 
{change_code}

code_change_explaination:
{code_change_explaination}

Please fill in the infomation in the given template below.

template:
if_API_misuse_fix: yes or no
reason_of_judgement: reason of the judgement
"""


with open("C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_v2_compare.json", encoding="utf-8") as f:
    data = f.readlines()
    data = [line for line in data if line != "\n"]
    data = [json.loads(line) for line in data]
    data = [line for line in data if line["manual_label"] != line["auto_label"]]

for idx in range(0, len(data)):
    print(idx, "/", len(data))
    manual_comments = data[idx]["manual_comments"]
    change = data[idx]["change"]
    number = data[idx]["number"]
    added = data[idx]["added"]
    removed = data[idx]["removed"]
    manual_label = data[idx]["manual_label"]
    code_change_explaination = data[idx]["code_change_explaination"]

    doc = ""
    doc += "manual_comments: \n"
    doc += manual_comments
    doc += "change: \n"
    doc += change
    doc += "added: \n"
    doc += added
    doc += "removed: \n"
    doc += removed
    doc += "manual_label: \n"
    doc += manual_label


    similar_example = ""
    # collection = get_VDB()
    # similar_example = collection.query(
    #     query_texts=[doc],
    #     n_results=1
    # )

    prompt_2 = template_2.format(change_code=change, code_change_explaination=code_change_explaination)
    print("prompt_2", len(prompt_2))

    output_2 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt_2}
        ]
    )

    output_2 = output_2["choices"][0]["message"]["content"]

    output_2_dict = {}
    try:
        for line in output_2.splitlines():
            if line != "":
                key, value = line.split(":")
                output_2_dict[key] = value
    except:
        pass

    output = {
        "number": number,
        "misuse_classification": output_2_dict
    }
    print(output)
    
    with open('C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_classification_v2_VSDB_diff_only_refine.json', 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
