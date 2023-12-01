""" LLMs for DeepShovel: 结构化数据抽取 - 三元组抽取（输入是预处理好的段落文件） """
# 分步骤提示大模型抽取三元组：1. 抽取实体；2. 实体关联到本体；3. 抽取关系
# 需要提供 <本体> 类型定义和 <关系> 结构规范
# 以mgkg的段落文本为例
import os
import openai
import json
import logging
import time
import random
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()
os.environ["http_proxy"] = "http://10.10.1.3:10000"
os.environ["https_proxy"] = "http://10.10.1.3:10000"
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = openai.api_key


# 调用API并使用重试机制处理rate limit error和其他异常
def get_completion(prompt, model="gpt-3.5-turbo"):
    for i in range(3):  # Retry the API call up to 3 times
        try:
            messages = [
                {"role": "system", "content": "I want you to act as a entity and relation extractor to help me build a medical knowledge graph from a paragraph."},
                {"role": "user", "content": prompt}
            ]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message["content"]
        except openai.error.RateLimitError:  # If rate limit is exceeded
            wait_time = (2 ** i) + random.random()  # Exponential backoff with jitter
            logging.warning(f"Rate limit exceeded. Retrying after {wait_time} seconds.")
            time.sleep(wait_time)  # Wait before retrying
        except Exception as e:  # If any other error occurs
            logging.error(f"API call failed: {str(e)}")
            return None  # Return None for failure
    logging.error("Failed to call OpenAI API after multiple retries due to rate limiting.")
    return None  # Return None for failure


# 从schema文件中获取本体（ontology - entity_label），20个
def get_entity_labels():
    entity_labels = []

    # 读取excel工作表MGKG_Schema_2023-05-05.xlsx - ontology
    df = pd.read_excel('../../mgkg/MGKG_Schema_2023-05-05.xlsx', sheet_name='ontology')
    # 按行迭代数据
    for index, row in df.iterrows():
        # 读取行中的每个单元格
        entity_label = row['schema']
        entity_labels.append(entity_label)

    return entity_labels


# 从schema文件中获取关系（relation），33个
def get_relations():
    relations = []

    # 读取excel工作表MGKG_Schema_2023-05-05.xlsx - relations
    df = pd.read_excel('../../mgkg/MGKG_Schema_2023-05-05.xlsx', sheet_name='relations')
    # 按行迭代数据
    for index, row in df.iterrows():
        # 读取行中的每个单元格
        relation_name = row['schema']
        relations.append(relation_name)

    return relations


def triple_extraction(paragraph: str, entity_labels: list, schema_relations: list):
    # 1. 抽取实体
    prompt1 = f"""
I will give you a paragraph. Extract as many named entities as possible from it. Your answer should only contain a list and nothing else. 
---
Here is an example:

paragraph: 
myasthenia gravis is characterized by muscle weakness. prednisolone is a treatment for myasthenia gravis.

your answer: 
[
    "myasthenia gravis",
    "muscle weakness",
    "prednisolone"
]
---
Here is the paragraph you should process:
{paragraph}
"""

    entity_list = get_completion(prompt1)
    
    # 2. 实体关联到本体
    prompt2 = f"""This is the entity list you have just generated:

{entity_list}

Classify every entity in into one of the categories in the following list. You should not classify any entity into a category that in not in the following list.

{entity_labels}

Your result should be a JSON dictionary with entities being the keys and categories being the values. There should be nothing in your answer except the JSON dictionary.
---
Here is an example:

paragraph: 
myasthenia gravis is characterized by muscle weakness. prednisolone is a treatment for myasthenia gravis.

entity list:
[
    "myasthenia gravis",
    "muscle weakness",
    "prednisolone"
]
your answer:
{{
    "myasthenia gravis": "disease",
    "muscle weakness": "symptom",
    "prednisolone": "medication"
}}
"""

    entity_category_dict = get_completion(prompt2)
    
    # 3. 抽取关系
    prompt3 = f"""
The following is the paragraph:

{paragraph}

The following is the "entity list" you have just generated:

{entity_list}

Extract as many relations as possible from the paragraph. Your result should be a list of triples and nothing else. 
The first and third element in each triple should be in the "entity list" you have generated and the second element should be in the following "relation category list". 
You should not extract any relation that the second element in it is not in the following "relation category list". 
The relation you choose should be precise and diverse. You shouldn't use "treatment" to describe all the relations.

Here is the "relation category list":
{schema_relations}

---
Here is an example:

paragraph: 
myasthenia gravis is characterized by muscle weakness. prednisolone is a treatment for myasthenia gravis.

entity list:
[
    "myasthenia gravis",
    "muscle weakness",
    "prednisolone"
]

your answer:
[
    ["myasthenia gravis", "presented with", "muscle weakness"],
    ["prednisolone", "treatment", "myasthenia gravis"],
]

"""

    relation_list = get_completion(prompt3)
    
    try:
        p_entity_list = json.loads(entity_list)
        p_entity_category_dict = json.loads(entity_category_dict)
        p_relation_list = json.loads(relation_list)
        # print("# JSON load successful!")
        load_flag = True
        return {
            "entity_list": p_entity_list,
            "entity_category_dict": p_entity_category_dict,
            "relation_list": p_relation_list
        }, load_flag
    except:
        # print("# JSON load failed!")
        load_flag = False
        return {
            "entity_list": entity_list,
            "entity_category_dict": entity_category_dict,
            "relation_list": relation_list
        }, load_flag


entity_labels = get_entity_labels()
schema_relations = get_relations()
print("# entity labels:\n", entity_labels)
print("-" * 100)
print("# schema_relations:\n", schema_relations)
print("-" * 120)


"""
paragraph_dict = {
    "paragraph": paragraph,
    "relations": [
        {
            "head": head,
            "head_label": head_label,
            "relation": relation,
            "tail": tail,
            "tail_label": tail_label
        },
        {
            "head": head,
            "head_label": head_label,
            "relation": relation,
            "tail": tail,
            "tail_label": tail_label
        },
        ...
    ]
}
"""
paragraph_file = "../data/mgkg_data/paragraph_pubmed_0720_test.txt"
result_write_file = "../results/mgkg_result/paragraph_pubmed_0720_test_result.json"
result_single_write_file = "../results/mgkg_result/paragraph_pubmed_0720_test_result_single.json"
retry_write_file = "../data/mgkg_data/paragraph_pubmed_0720_test_retry.txt"

# paragraph_file = "../data/mgkg_data/paragraph_pubmed_0720.txt"
# result_write_file = "../results/mgkg_result/paragraph_pubmed_0720_result.json"
# result_single_write_file = "../results/mgkg_result/paragraph_pubmed_0720_result_single.json"
# retry_write_file = "../data/mgkg_data/paragraph_pubmed_0720_retry.txt"
with open(paragraph_file, "r") as file:
    paragraphs = file.readlines()

retry_paragraphs = []   # JSON load失败的paragraph
paragraph_dict_list = []    # 所有paragraph的三元组抽取结果
success_cnt = 0
fail_cnt = 0

# 将抽取结果paragraph_dict持续性写入result_single_write_file文件
with open(result_single_write_file, "w", newline='\n') as wrt_single_file:
    for paragraph in tqdm(paragraphs, total=len(paragraphs), desc='Processing paragraphs'):
        result, load_flag = triple_extraction(paragraph, entity_labels, schema_relations)

        # 折中方案：JSON load失败，重试
        for i in range(3):
            if not load_flag:
                print("# JSON load failed! Retry...")
                time.sleep(5)  # Wait before retrying
                result, load_flag = triple_extraction(paragraph, entity_labels, schema_relations)
            else:
                break

        if load_flag:   # JSON load成功
            success_cnt += 1
            entities = result['entity_list']
            entity_labels = result['entity_category_dict']
            relations = result['relation_list']
            relation_dict_list = []  # 每个paragraph抽取的的所有relation
            for item in relations:
                if len(item) != 3:
                    continue
                head = item[0]
                relation = item[1]
                tail = item[2]
                head_lebal = ""
                tail_label = ""
                # print(f'{head}, {relation}, {tail}')
                entity_keys = entity_labels.keys()
                for key in entity_keys:
                    if key in head:
                        head_label = entity_labels[key]
                    if key in tail:
                        tail_label = entity_labels[key]
                relation_dict = {   # paragraph中抽取的一个relation
                    "head": head,
                    "head_label": head_label,
                    "relation": relation,
                    "tail": tail,
                    "tail_label": tail_label
                }
                relation_dict_list.append(relation_dict)
            paragraph_dict = {
                "paragraph": paragraph,
                "relations": relation_dict_list
            }
            paragraph_dict_list.append(paragraph_dict)
        else:   # JSON load失败
            fail_cnt += 1
            retry_paragraphs.append(paragraph)
            paragraph_dict = {
                "paragraph": paragraph,
                "relations": "failure"
            }
            paragraph_dict_list.append(paragraph_dict)
        json.dump(paragraph_dict, wrt_single_file, indent=4)
        wrt_single_file.write('\n')  # 添加换行符
        print("# paragraph_cnt {}, success_cnt {}, fail_cnt {}".format(len(paragraphs), success_cnt, fail_cnt))


print("# paragraph_cnt:", len(paragraphs))
print("# success_cnt:", success_cnt)
print("# fail_cnt:", fail_cnt)

# 将抽取结果paragraph_dict_list一次性写入文件
with open(result_write_file, "w") as json_file:
    json.dump(paragraph_dict_list, json_file, indent=4)

# 将JSON load失败的paragraph一次性写入新文件
with open(retry_write_file, 'w') as retry_file:
    for item in retry_paragraphs:
        retry_file.write("%s" % item)
