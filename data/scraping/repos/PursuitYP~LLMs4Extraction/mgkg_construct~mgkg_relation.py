""" MGKG: 关系抽取处理（三元组清洗、实体链接、关系对齐） """
import os
import openai
from dotenv import load_dotenv
import os
import regex as re
from tqdm import tqdm
import pandas as pd
import csv
import random
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging


# Use gpt-3.5-turbo with designed prompt to extract data
def extract_chunk(document, template_prompt):
    
    prompt=template_prompt.replace('<document>', document)

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo', 
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return "1. " + response['choices'][0]['message']['content']


# 调用API并使用重试机制处理rate limit error和其他异常
def get_completion(prompt, model="gpt-3.5-turbo"):
    for i in range(5):  # Retry the API call up to 5 times
        try:
            messages = [{"role": "user", "content": prompt}]
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


# 处理过程信息写入log文件
def log_to_file(log_file, message):
    try:
        with open(log_file, 'a') as file:
            file.write(message + '\n')
    except Exception as e:
        logging.error(f'Failed to log to file {log_file}: {str(e)}')
        raise


# 从schema文件中获取本体（ontology - entity_label），20个
def get_entity_labels():
    entity_labels = []

    # 读取excel工作表MGKG_Schema_2023-05-05.xlsx - ontology
    df = pd.read_excel('../mgkg/MGKG_Schema_2023-05-05.xlsx', sheet_name='ontology')
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
    df = pd.read_excel('../mgkg/MGKG_Schema_2023-05-05.xlsx', sheet_name='relations')
    # 按行迭代数据
    for index, row in df.iterrows():
        # 读取行中的每个单元格
        relation_name = row['schema']
        relations.append(relation_name)

    return relations


# 实体链接（实体 -> 本体）
def entity_linking(sentence, entity, entity_labels):
    sentence_example = "myasthenia gravis is characterized by muscle weakness."
    entity_example = "myasthenia gravis"
    entity_label_example = "disease"

    sentence_example_1 = "prednisolone is a treatment for myasthenia gravis."
    entity_example_1 = "prednisolone"
    entity_label_example_1 = "medication"

    entity_prompt = f'''You will read a "sentence" and an "entity" extracted from that sentence.
Please classify the entity into one of the following "categories" based on the contextual information of this entity in the sentence.
If you think the entity does not belong to any of the following categories, output \"none\".
Your output can only be one of these categories or \"none\".
---
Here is an example:

SENTENCE
{sentence_example}
ENTITY
{entity_example}
CATEGORIES
{entity_labels}
OUTPUT
{entity_label_example}
---
Here is another example:

SENTENCE
{sentence_example_1}
ENTITY
{entity_example_1}
CATEGORIES
{entity_labels}
OUTPUT
{entity_label_example_1}
---
Here is the entity you need to categorize and the sentence from which this entity was extracted:

SENTENCE
{sentence}
ENTITY
{entity}
CATEGORIES
{entity_labels}
OUTPUT

'''
    # print(entity_prompt)
    # print("-" * 100)
    entity_label = get_completion(entity_prompt)
    return entity_label


# 关系对齐（关系 -> schema关系）
def relation_alignment(sentence, head, tail, relation, schema_relations):
    sentence_example = "We present a case of bilateral multifocal central serous chorioretinopathy in a 40-year-old male who suffered from myasthenia gravis and was receiving oral prednisolone ."
    relations_example = [0.95, 'a 40-year-old male', 'was receiving', 'oral prednisolone']
    relation_aligned_example = "use of drug"

    sentence_example_1 = "Lately it has been suggested that reducing the light dose fluence is safer because standard PDT may cause severe choroidal ischaemia 9 while low-fluence PDT minimizes the risk of Bruchs membrane rupture ."
    relations_example_1 = [0.95, 'standard PDT', 'may cause', 'severe choroidal ischaemia 9']
    relation_aligned_example_1 = "caution"

    triple = [head, relation, tail]

    relation_prompt = f'''You will read a 'sentence' and a 'triple' (\[head entity, 'relation', tail entity\]) extracted from that sentence.
Please align this "relation" into one of the following "relation schemas" based on the contextual information of that relation in the sentence and in the triple.
If you believe that the relation cannot be aligned to any of the following relation schemas, output \"none\".
Your output can only be one of the relation schemas or \"none\".
---
Here is an example:

SENTENCE
{sentence_example}
TRIPLE
{relations_example[1:]}
RELATION
{relations_example[2]}
RELATION SCHEMAS
{schema_relations}
OUTPUT
{relation_aligned_example}
---
Here is another example:

SENTENCE
{sentence_example_1}
TRIPLE
{relations_example_1[1:]}
RELATION
{relations_example_1[2]}
RELATION SCHEMAS
{schema_relations}
OUTPUT
{relation_aligned_example_1}
---
Here is the relation you need to align and the sentence from which the relation is extracted:

SENTENCE
{sentence}
TRIPLE
{triple}
RELATION
{relation}
RELATION SCHEMAS
{schema_relations}
OUTPUT

'''
    # print(relation_prompt)
    # print("-" * 100)
    relation_aligned = get_completion(relation_prompt)
    return relation_aligned


# 根据关系的上下文句子，利用LLMs进行关系对齐和实体链接
def process_relations(sentence, relations, entity_labels, schema_relations):
    new_relations = []

    for relation_info in relations:
        head = relation_info[1]
        relation = relation_info[2]
        tail = relation_info[3]
        # 1. 根据entity_labels进行实体链接
        head_label = entity_linking(sentence, head, entity_labels)
        tail_label = entity_linking(sentence, tail, entity_labels)
        # 如果head_label或者tail_label为none，直接continue
        if head_label == "none" or tail_label == "none":
            continue

        # 2. 根据schema_relations进行关系对齐
        relation_aligned = relation_alignment(sentence, head, tail, relation, schema_relations)
        # 如果relation_aligned为none，直接continue
        if relation_aligned == "none":
            continue

        # 3. 设置筛选条件，更新new_relations
        new_relation = [head, head_label, relation_aligned, relation, tail, tail_label]
        new_relations.append(new_relation)

    return new_relations


# 从txt文件中读取关系，并进行预处理筛除一些项
def get_relations_from_txt(input_rel_file):
    sentence_relations_pack = []    # 保存每个句子和从中抽取到的多个关系
    print("# new relation schema: [head, head_label, relation_aligned, old_relation, tail, tail_label]")
    # 打开待处理的rel_file，读取数据，进行实体链接和关系对齐
    with open(input_rel_file, 'r') as f:
        lines = f.readlines()
        cnt = 0
        sentence = ""
        relations = []
        for line in tqdm(lines, desc='Processing sentence lines'):
            # 移除句子结尾的 "\n"
            line = line[:-1]
            cnt += 1
            # if cnt <= 1000:
            if True:
                # print(line)
                # 如果遇到空行，说明一个句子和关系结束了
                if line == "":
                    # 打印输出，跳过没有抽取到关系的句子
                    if relations != []:
                        sentence_relations_item = [sentence, relations]
                        sentence_relations_pack.append(sentence_relations_item)
                    # 重置sentence和relations
                    sentence = ""
                    relations = []
                # 否则，说明该行是句子或者抽取的关系
                else:
                    # 如果sentence为空，说明该句是sentence
                    if sentence == "":
                        sentence = line
                    # 否则，该句为关系，提取 [confidence, head, relation, tail] ，加到relations
                    else:
                        confidence = float(line[0:4])
                        relation = re.findall(";.*;", line)[0].strip(';').strip(';').strip()
                        head = re.findall("\(.*;", line)[0].strip('\(').strip(';').strip()
                        head = head.replace(relation, "")[:-2]
                        tail = re.findall(";.*\)", line)[0].strip(';').strip('\)').strip()
                        tail = tail.replace(relation, "")[2:]
                        # 三元组不完整，跳过
                        if head == "" or relation == "" or tail == "":
                            continue
                        # 三元组中存在stopwords，跳过
                        if head.lower() in stopwords_ls or relation.lower() in stopwords_ls or tail.lower() in stopwords_ls:
                            continue
                        # 实体太长（>6），跳过
                        if len(head.split()) > 6 or len(tail.split()) > 6:
                            continue
                        # confidence分数小于0.6，跳过
                        if confidence < 0.6:
                            continue
                        # 如果动物实验相关名词在实体名中出现，跳过
                        animal_flag = False
                        for key in exp_animals_keys:
                            if key.lower() in head.lower() or key.lower() in tail.lower():
                                animal_flag = True
                                break
                        if animal_flag:
                            continue
                        relation_info = [confidence, head, relation, tail]
                        relations.append(relation_info)
                        
    return sentence_relations_pack


def main(input_rel_file, write_csv_file):
    # 从txt文件中读取关系，并进行预处理筛除一些项
    sentence_relations_pack = get_relations_from_txt(input_rel_file)

    # 新关系结构：[head, head_label, relation_aligned, old_relation, tail, tail_label]
    final_relations = []    # 处理后最终的关系：三元组清洗、实体链接、关系对齐
    with open(write_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入csv表头
        writer.writerow(["head", "head_label", "relation", "old_relation", "tail", "tail_label"])
        # 限制最大线程数为4，缓解rate_limit_exceeded报错
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 多线程处理
            futures = {executor.submit(process_relations, item[0], item[1], entity_labels, schema_relations): item for item in sentence_relations_pack}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing relations'):
                # 收集完成的线程处理好的结果
                new_relations = future.result()
                if new_relations is None:
                    log_to_file(log_file, f'Failed to process item {futures[future]}')
                else:
                    for item in new_relations:
                        # 边处理边写入，写入处理好的新关系
                        writer.writerow(item)
                        final_relations.append(item)
                    log_to_file(log_file, f'Successfully processed item!')


if __name__ == "__main__":
    load_dotenv()
    os.environ["http_proxy"] = "http://10.10.1.3:10000"
    os.environ["https_proxy"] = "http://10.10.1.3:10000"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    os.environ['OPENAI_API_KEY'] = openai.api_key

    # 打开并读取stopwords
    with open('stopwords.txt', 'r') as f:
        stopwords_f = f.readlines()
    stopwords_ls = []
    for w in stopwords_f:
        stopwords_ls.append(w.strip().strip('\n'))

    # 获取并打印entity_labels和schema_relations
    entity_labels = get_entity_labels()
    schema_relations = get_relations()
    print("# entity labels and schema relations: ")
    print(len(entity_labels), entity_labels)
    print(len(schema_relations), schema_relations)
    print("-" * 100)

    # 动物实验关键词和指代不明常见词
    exp_animals_keys = ['experimental autoimmune myasthenia gravis', 'EAMG', 'dog', 'mice', 'animal', 'guinea pig', 'rats', 'vertebrate']
    ref_ubclear_keys = ['patient', 'medication', 'drug', 'test', 'gene', 'symptom', 'scale']

    # input_rel_file = "../../projects/openie6/predictions.txt"
    # write_csv_file = "../../projects/openie6/predictions_new.csv"
    # input_rel_file = "../../projects/openie6/rel_results/rel_abs_pubmed_selected_clinical.txt"
    # write_csv_file = "../../projects/openie6/rel_results/rel_abs_pubmed_selected_clinical_new.csv"
    # input_rel_file = "../../projects/openie6/rel_results/rel_abs_1_10001.txt"
    # write_csv_file = "../../projects/openie6/rel_results/rel_abs_1_10001_new.csv"
    # input_rel_file = "../../projects/openie6/rel_results/rel_abs_90001_100001.txt"
    # write_csv_file = "../../projects/openie6/rel_results/rel_abs_90001_100001_new.csv"
    input_rel_file = "../../projects/openie6/rel_results/rel_display.txt"
    write_csv_file = "../../projects/openie6/rel_results/rel_display_new.csv"
    print(f"write_csv_file: {write_csv_file}")
    print("-" * 100)
    log_file = 'log_mgkg.txt'

    main(input_rel_file, write_csv_file)
