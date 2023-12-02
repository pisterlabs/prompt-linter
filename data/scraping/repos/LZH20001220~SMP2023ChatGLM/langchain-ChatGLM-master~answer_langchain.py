from configs.model_config import *
from chains.local_doc_qa import LocalDocQA
import os
import nltk
from models.loader.args import parser
import models.shared as shared
from models.loader import LoaderCheckPoint
from pypinyin import pinyin, Style
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from tqdm import tqdm
import os
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


import json

# Show reply with source text from input document
REPLY_WITH_SOURCE = False

def chinese_to_english(input_text):
    pinyin_list = pinyin(input_text, style=Style.NORMAL)
    english_text = ''.join([p[0] for p in pinyin_list])
    return english_text.lower()  # 将拼音转换为小写英文字符

# 从JSON文件中读取问题数据
def read_questions_from_json(json_file_path):
    questions = []
    with open(json_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            question = json.loads(line)
            questions.append(question)
    return questions

def check_type3_1(question):
    if "法定代表人" in question:
        return False
    if "简要" in question or "情况" in question:
        return True
    else:
        return False
    
    
# 将问题和回答保存为JSON文件
def save_answers_to_json(answers, json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as file:
        for answer in answers:    
            json.dump(answer, file, ensure_ascii=False, indent=None)
            print("", file=file)

if __name__ == '__main__':
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.history_len = LLM_HISTORY_LEN

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(llm_model=llm_model_ins,
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          top_k=VECTOR_SEARCH_TOP_K)
    
    #中文Wikipedia数据导入示例：
    embedding_model_name = "GanymedeNil/text2vec-large-chinese"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    test_questions = read_questions_from_json("../dataset/match_test.json")
    # 读取文件夹中的所有txt文件并生成向量文件

    answers = []
    for item in test_questions:
        # torch.cuda.empty_cache()
        question_id = item["id"]
        question_text = item["question"]
        file_name = item["filename1"]
        file_path = ("../data/alltxt/" + file_name)
        
        flag = check_type3_1(question_text)
        if flag == False:
            continue
        
        if file_name == "":
            continue
        
        output_text = ""
        
        txt_file_name = os.path.splitext(os.path.basename(file_name))[0]
        txt_file_name = chinese_to_english(txt_file_name)
        
        vs_path = ("../dataset/vector_files/" + txt_file_name + "/")
        if not os.path.exists(vs_path):
            continue
        print(vs_path) 
        history = []  
        query = question_text
        last_print_len = 0
        for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
                                                                    vs_path=vs_path,
                                                                    chat_history=history,
                                                                    ):
            if STREAMING:
                # print(resp["result"][last_print_len:], end="", flush=True)
                output_text = resp["result"].strip()
                output_text = output_text.replace("\n", "")
                # print(output_text, end="", flush=True)
                last_print_len = len(resp["result"])
                # print(resp["result"])
                
        answer_dict = {
            "id": question_id,
            "question": question_text,
            "answer": output_text
        }
        answers.append(answer_dict)
        
        save_answers_to_json(answers, "../dataset/answers_lang.json")
  
