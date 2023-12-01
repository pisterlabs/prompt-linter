import json
import os.path
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

# 中文Wikipedia数据导入示例：
embedding_model_name = 'WangZeJun/simbert-base-chinese'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

test_questions = open("test_questions_keyword.jsonl").readlines()
question_2020 = []
for test_question in test_questions:
    question = json.loads(test_question)
    # if "2020" in question:
    question_2020.append(question)

file_form = {}
first_label = json.load(open("form_list_keywords.json", "r"))
text_label = json.load(open("text_list_keywords.json", "r"))
from transformers import AutoTokenizer, AutoModel
import json
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


import re


def re_chinese(a_str):
    a_list = re.findall(r'[^\x00-\xff]', a_str)
    return "".join(a_list)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, device='cuda')
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
    # model = model.eval()
    # from fastllm_pytools import llm
    # model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    #
    # model = llm.from_hf(model, tokenizer, dtype="float16")
    result = []
    for question_one in question_2020:
        all_texts = []
        year_prefix = question_one['year']
        stock_name_list = question_one['stock_name']
        rows = []
        # jingzhunpipeibiaoge
        if "keyword" in question_one:
            # print(question_one)
            for stock_name_one in stock_name_list:
                if 'match_annoy_name' in question_one:
                    if len(question_one['match_annoy_name']):
                        for stock_form in question_one['match_annoy_name']:
                            stock_form_key = os.path.join("./alltext/alltxt", stock_form)
                            for question_one_keyword in question_one['keyword']:
                                if stock_form_key in first_label:
                                    for one in first_label[stock_form_key]:
                                        if question_one_keyword in one:
                                            rows.append(one)
                if len(rows) == 0 and 'search_stock_name' in question_one:
                    for search_stock_name in question_one['search_stock_name']:
                        stock_form_key = os.path.join("./alltext/alltxt", search_stock_name)
                        for question_one_keyword in question_one['keyword']:
                            if stock_form_key in first_label:
                                for one in first_label[stock_form_key]:
                                    if question_one_keyword in one:
                                        rows.append(one)
            if len(rows):
                message = "\n".join(rows)
                all_texts = [
                    "[Round 0]\n 根据以下几个表格:\n" + message[:1000] + " 解决问题：\n" + question_one['question']
                    + "    \n答：",
                ]
            if len(all_texts) == 0:

                for stock_name_one in stock_name_list:
                    if 'match_annoy_name' in question_one:
                        if len(question_one['match_annoy_name']):
                            for stock_form in question_one['match_annoy_name']:
                                stock_form_key = os.path.join("./alltext/alltxt", stock_form)
                                for question_one_keyword in question_one['keyword']:
                                    if stock_form_key in text_label:
                                        for one in text_label[stock_form_key]:
                                            if question_one_keyword in one:
                                                rows.append(one)
                    if 'search_stock_name' in question_one:
                        for search_stock_name in question_one['search_stock_name']:

                            stock_form_key = os.path.join("./alltext/alltxt", search_stock_name)
                            for question_one_keyword in question_one['keyword']:
                                if stock_form_key in text_label:
                                    for one in text_label[stock_form_key]:
                                        if question_one_keyword in one:
                                            rows.append(one)
                if len(rows):
                    message = "\n".join(rows)
                    all_texts = [
                        "[Round 0]\n 根据以下几段文字:\n" + message[:1000] + " 解决问题：\n" + question_one['question']
                        + "    \n答：",
                    ]
            # print(all_texts)
        if len(all_texts) == 0:
            idx = 0
            docs = []
            chinese_row_mapping = {}
            annoy_form_data = []
            for stock_name_one in stock_name_list:
                if 'match_annoy_name' in question_one:
                    if len(question_one['match_annoy_name']):
                        for stock_form in question_one['match_annoy_name']:
                            stock_form_key = os.path.join("./alltext/alltxt", stock_form)
                            if stock_form_key in first_label:
                                for one in first_label[stock_form_key]:
                                    annoy_form_data.append(one)
                if 'search_stock_name' in question_one:
                    if len(question_one['search_stock_name']):
                        for stock_form in question_one['search_stock_name']:
                            stock_form_key = os.path.join("./alltext/alltxt", stock_form)
                            if stock_form_key in first_label:
                                for one in first_label[stock_form_key]:
                                    annoy_form_data.append(one)
            annoy_form_data = list(set(annoy_form_data))
            for row in annoy_form_data:
                metadata = {"source": f'doc_id_{idx}'}
                idx += 1
                if isinstance(row, str):
                    chinese_row = re_chinese(row)
                    for stock_name_one in stock_name_list:
                        chinese_row = chinese_row.replace(stock_name_one, "")
                    chinese_row_mapping[chinese_row] = row
                    docs.append(Document(page_content=chinese_row, metadata=metadata))
            if len(docs):
                vector_store = FAISS.from_documents(docs, embeddings)

                query = question_one['question']
                for stock_name_one in stock_name_list:
                    query = query.replace(stock_name_one, "")
                for stock_name_one in question_one['company']:
                    query = query.replace(stock_name_one, "")
                for stock_name_one in year_prefix:
                    query = query.replace(stock_name_one, "")
                file_form_one = vector_store.similarity_search(query)
                chinese_row_prompt = "下一个表格\n".join(
                    [chinese_row_mapping[file_form_one.page_content] for file_form_one in
                     file_form_one[:3]])
                all_texts = [
                    "[Round 0]\n 根据以下几个表格:\n" + chinese_row_prompt[:1000] + " 寻找解决问题的方法：" +
                    question_one['question'] + "    \n答：",
                ]
        if len(all_texts) == 0:
            idx = 0
            docs = []
            chinese_row_mapping = {}
            annoy_form_data = []
            for stock_name_one in stock_name_list:
                if 'match_annoy_name' in question_one:
                    if len(question_one['match_annoy_name']):
                        for stock_form in question_one['match_annoy_name']:
                            stock_form_key = os.path.join("./alltext/alltxt", stock_form)
                            if stock_form_key in text_label:
                                for one in text_label[stock_form_key]:
                                    annoy_form_data.append(one)
                if 'search_stock_name' in question_one:
                    if len(question_one['search_stock_name']):
                        for stock_form in question_one['search_stock_name']:
                            stock_form_key = os.path.join("./alltext/alltxt", stock_form)
                            if stock_form_key in text_label:
                                for one in text_label[stock_form_key]:
                                    annoy_form_data.append(one)
            annoy_form_data = list(set(annoy_form_data))
            for row in annoy_form_data:
                metadata = {"source": f'doc_id_{idx}'}
                idx += 1
                if isinstance(row, str):
                    chinese_row = re_chinese(row)
                    for stock_name_one in stock_name_list:
                        chinese_row = chinese_row.replace(stock_name_one, "")
                    chinese_row_mapping[chinese_row] = row
                    docs.append(Document(page_content=chinese_row, metadata=metadata))
            if len(docs):
                vector_store = FAISS.from_documents(docs, embeddings)

                query = question_one['question']
                for stock_name_one in question_one['company']:
                    query = query.replace(stock_name_one, "")
                for stock_name_one in stock_name_list:
                    query = query.replace(stock_name_one, "")
                for stock_name_one in year_prefix:
                    query = query.replace(stock_name_one, "")
                file_form_one = vector_store.similarity_search(query)
                chinese_row_prompt = "下一段文字\n".join(
                    [chinese_row_mapping[file_form_one.page_content] for file_form_one in
                     file_form_one[:3]])
                all_texts = [
                    "[Round 0]\n 根据以下几段文字:\n" + chinese_row_prompt[:1000] + " 寻找解决问题的方法：" +
                    question_one['question'] + "    \n答：",
                ]
        if len(all_texts) == 0:
            all_texts = [
                "[Round 0]\n问：" + question_one['question'] + "    \n答：",
            ]
        response, history = model.chat(tokenizer,
                                       all_texts[0],
                                       history=[],
                                       max_length=4096,
                                       top_p=0.7,
                                       temperature=0.95)
        print(all_texts[0])
        print(response)
        question_one['answer'] = response
        result.append(question_one)
    d = []
    for i in result:
        d.append(json.dumps(i))
    open("keyword_mapping&search_gen.json", "w").write("\n".join(d))
