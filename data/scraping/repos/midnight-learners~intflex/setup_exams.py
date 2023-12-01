import configparser
from typing import Tuple, List, Dict, Any
import openai
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
import yaml

from db import PostgresqlClient


def parse_raw_questions_to_json(raw_questions: str) -> list[dict]:
    """
    Parse ChatGPT generated string content, and convert each generated question --to--> json format
    :param raw_questions: string of ChatGPT response
    :return: list of question in json format
    """
    processed_question = []
    for raw_question in raw_questions.split('<DIVIDED>'):
        raw_question = raw_question.split('\n')
        json_question = {}
        for section in raw_question:
            section = section.replace(" ", "")
            if section.startswith("question"):
                question_content = section.split("question:")[1]
                json_question["question"] = question_content

            elif section.startswith("options"):
                json_options = {}
                for option in section.split("options:")[1].split("|"):
                    option_tag = option.split(":")[0]
                    option_content = option.split(":")[1]
                    json_options[option_tag] = option_content
                json_question["options"] = json_options

            elif section.startswith("answer"):
                answer_content = section.split("answer:")[1]
                json_question["answer"] = answer_content

        if len(json_question) == 3:
            processed_question.append(json_question)

    return processed_question


def split_article(article: str, split_length: int = 1000) -> list[str]:
    split_text_list = []
    text_length = len(article)
    for i in range(0, text_length, split_length):
        if i + 2 * split_length >= text_length - 1:
            split_text_list.append(article[i:])
            break
        else:
            split_text_list.append(article[i:i + split_length])

    return split_text_list


def generate_knowledge(article: str, num_knowledge_point: int) -> str:
    """
    Generate knowledge points based on article
    """
    prompt_1 = """
    You have been assigned the responsibility of overseeing staff training and there is a company article available for your reference. \
    Your task is to analysis the article and extract {NUM_KNOWLEDGE_POINT} key knowledge mentioned in the article that you believe are crucial for employees, \
    and describe each knowledge in a comprehensive summary in Chinese with about 100 words. \
    Please mark each knowledge clearly with bullet point.
    
    Output example: ```
    -- <your generated knowledge name> | <your generated knowledge summary>
    ```
    Article: ```{ARTICLE}```
    """.format(NUM_KNOWLEDGE_POINT=num_knowledge_point, ARTICLE=article)

    start_time = time.time()
    response_gpt = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt_1,
        temperature=0.7,
        max_tokens=2048,
        n=1,
        top_p=1,
    )
    extracted_knowledge_points = response_gpt.choices[0].text
    print(f"\nopenai completion time cost: {time.time() - start_time}")
    return extracted_knowledge_points


def generate_question(knowledge: str, num_question: int):
    """
    Generate question based on article
    """

    """Generate question base on the knowledge point"""
    prompt_2 = """
    You are the person in charge of employee orientation, and you have a list of knowledge points. Your task is to \
    generate multiple-choice question for each knowledge point to see how well the employee has mastered the knowledge \
    point. Each multiple-choice question has 1 correct option and 3 interference options (must ensure interference \
    options are wrong). 
    Now, Please generate {NUM_QUESTION} question and output each question in the format of following example below. \
    Make sure generated content is Chinese, adding <DIVIDED> delimiter tag at the end of each question.\
    Output Example:
    ```
    question: <your generated question>
    options: A: <Option A content> | B: <Option B content> | C: <Option C content> | D: <Option D content>
    answer: <The correct option of your generated question>
    <DIVIDED>
    ```

    Knowledge points: ```{KNOWLEDGE_POINTS}```
    """.format(NUM_QUESTION=num_question, KNOWLEDGE_POINTS=knowledge)

    start_time = time.time()
    response_gpt = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt_2,
        temperature=0.9,
        max_tokens=2048,
        n=1,
    )
    raw_questions = response_gpt.choices[0].text
    print(f"\nopenai completion time cost: {time.time() - start_time}")

    return parse_raw_questions_to_json(raw_questions)


def import_database():
    """
    Import generated knowledge and questions into database
    """
    try:
        pg_config = yaml.safe_load(open("config.yaml", "r"))['postgresql']
        pg_client = PostgresqlClient(
            host=pg_config['host'],
            port=pg_config['port'],
            database=pg_config['database'],
            user=pg_config['user'],
            password=pg_config['password']
        )
    except Exception as e:
        print(f"Postgresql Client 初始化失败. Exception: {e}")
        return

    """读取SQL文件"""
    SQL_CONFIG = configparser.ConfigParser()
    SQL_CONFIG.read('sql.ini')

    """写入knowledge"""
    sql_template = SQL_CONFIG.get('insert', 'insert_knowledge', raw=True)
    knowledge_tuple_list = []
    with open('ZC-S-H-002知识点.txt', 'r', encoding='utf-8') as f:
        doc_code, knowledge_id = "ZC-S-H-002", 1
        for line in f:
            if line.strip():  # 如果不是空行
                knowledge_tuple_list.append((doc_code, knowledge_id, line.strip()))
                knowledge_id += 1

    # pg_client.execute_insert_many(sql_template, knowledge_data)

    """写入question"""
    sql_template = SQL_CONFIG.get('insert', 'insert_question', raw=True)
    with open('ZC-S-H-002题目.json', 'r', encoding='utf-8') as f:
        question_data = json.load(f)

    question_tuple_list = []
    for question_id, item in enumerate(question_data):
        question_tuple_list.append((doc_code, question_id + 1, item['question'], 0, json.dumps(item['options']), [item['answer']]))

    # pg_client.execute_insert_many(sql_template, question_tuple_list)
    pg_client.close()


if __name__ == "__main__":
    import_database()




    """Openai Config"""
    # load_dotenv()
    # openai.api_key = os.environ.get("OPENAI_API_KEY")
    # openai.proxy = "http://127.0.0.1:7890"

    """
    Generate knowledge points
    """
    # with open("../document_txt/人力资源部/ZC-S-H-002RBA管理手册（A1）.txt", "r") as f:
    #     txt = f.read()
    #
    # text_list = split_article(article=txt, split_length=600)
    # knowledge_text = ""
    # for text in tqdm(text_list, desc="Processing Texts"):
    #     knowledge_text += generate_knowledge(article=text, num_knowledge_point=2)
    #
    # knowledge_text = knowledge_text.replace("--", "\n").replace("\n\n", "\n").replace("\n\n", "\n").strip()
    #
    # with open('ZC-S-H-002知识点.txt', 'w', encoding='utf-8') as txt_f:
    #     txt_f.write(knowledge_text)

    """
    Generate questions
    """
    # with open('ZC-S-H-002知识点.txt', 'r', encoding='utf-8') as txt_f:
    #     knowledge_text = txt_f.read()
    #
    # knowledge_list = knowledge_text.split("\n")
    # json_question_list = []
    # for text in tqdm(knowledge_list, desc="Processing Knowledge Points"):
    #     question_json = generate_question(knowledge=text, num_question=1)
    #     print(question_json)
    #     json_question_list += question_json
    #
    # with open('ZC-S-H-002题目.json', 'w', encoding='utf-8') as json_f:
    #     json.dump(json_question_list, json_f, ensure_ascii=False, indent=4)
