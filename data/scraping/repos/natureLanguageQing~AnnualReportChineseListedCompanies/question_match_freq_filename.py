import pandas as pd
import json
import os

most_common = pd.read_csv('most_common.csv')
stock_mapping = json.load(open("stock_mapping.json", "r"))
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
import re

# 中文Wikipedia数据导入示例：
embedding_model_name = 'WangZeJun/simbert-base-chinese'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
all_txt_filename = os.listdir("alltext/alltxt")


def re_chinese(a_str):
    a_list = re.findall(r'[^\x00-\xff]', a_str)
    return "".join(a_list)


key_filename = []
filename_mapping = {}
idx = 0
docs = []
for row in all_txt_filename:
    metadata = {"source": f'doc_id_{idx}'}
    idx += 1
    if isinstance(row, str):
        ch_row = re_chinese(row)
        filename_mapping[ch_row] = row
        docs.append(Document(page_content=ch_row, metadata=metadata))
vector_store = FAISS.from_documents(docs, embeddings)
keyword_list = []
d = []
test_questions = open("question_center/uie_company.json").readlines()
for test_question in test_questions:
    test_question = json.loads(test_question)
    test_question["keyword"] = []
    test_question["stock_name"] = []
    test_question["match_annoy_name"] = []
    test_question["search_stock_name"] = []
    year_prefix = []
    if "2019年" in test_question['question']:
        year_prefix.append("2019年")
    if "2020年" in test_question['question']:
        year_prefix.append("2020年")
    if "2021年" in test_question['question']:
        year_prefix.append("2021年")
    test_question["year"] = year_prefix
    for most_common_one in most_common.values.tolist()[:2000]:
        if isinstance(most_common_one[1], str):
            if len(most_common_one[1]) > 2:
                if most_common_one[1] in test_question['question']:
                    print(most_common_one)
                    test_question["keyword"].append(most_common_one[1])
                    keyword_list.append(most_common_one[1])
    for stock_code, stock_name in stock_mapping.items():
        if isinstance(stock_name, str):
            if stock_name in test_question['question']:
                test_question["stock_name"].append(stock_name)
                for row in all_txt_filename:
                    for stock_name_one in test_question["stock_name"]:
                        if stock_name_one in row:
                            for year_prefix_one in year_prefix:
                                if year_prefix_one in row:
                                    test_question["match_annoy_name"].append(row)
                                    key_filename.append(row)
                    for company in test_question['company']:
                        if company in row:
                            for year_prefix_one in year_prefix:
                                if year_prefix_one in row:
                                    test_question["match_annoy_name"].append(row)
                                    key_filename.append(row)
    if len(test_question["match_annoy_name"]) == 0:
        file_form_one = vector_store.similarity_search(test_question['question'])
        for file_form_one in file_form_one:
            for year_prefix_one in year_prefix:
                if year_prefix_one in filename_mapping[file_form_one.page_content]:
                    test_question["search_stock_name"].append(filename_mapping[file_form_one.page_content])
                    key_filename.append(filename_mapping[file_form_one.page_content])
        for company in test_question['company']:

            file_form_one = vector_store.similarity_search(company)
            for file_form_one in file_form_one:
                for year_prefix_one in year_prefix:
                    if year_prefix_one in filename_mapping[file_form_one.page_content]:
                        test_question["search_stock_name"].append(filename_mapping[file_form_one.page_content])
                        key_filename.append(filename_mapping[file_form_one.page_content])
    d.append(test_question)
    print(json.dumps(test_question, ensure_ascii=False))
test_questions_keyword = open("test_questions_keyword.jsonl", "w")
for test_question in d:
    test_questions_keyword.write(json.dumps(test_question, ensure_ascii=False) + "\n")
keyword_list = list(set(keyword_list))
json.dump(keyword_list, open("keyword_list.json", "w"), ensure_ascii=False)
json.dump(key_filename, open("key_filename.json", "w"), ensure_ascii=False)
