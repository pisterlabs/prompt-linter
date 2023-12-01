import os
import ijson
import json
import random
import logging
from collections import defaultdict

from tqdm import tqdm
import sqlglot

from language_model import OpenAIModel

cur_dir = os.path.dirname(__file__)

logging.basicConfig(format='%(message)s\n',
                    level=logging.ERROR,
                    filename=os.path.join(cur_dir, 'sqlGrammerTest.log'),
                    filemode='w')

with open('extractedQuestion.json', 'r') as json_file:
    questions = json.load(json_file)

with open(f'extractedDB.json', 'r') as f:
    extractedDB = json.load(f)

prompt_head = "Here are Mysql tables, with their properties:\n\n"

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
api_key = os.environ.get('OPENAI_API_KEY')
print(api_key)
gpt3 = OpenAIModel(model="text-davinci-003", prompt_template=prompt_head, api_key=api_key, temperature=0.)

for db_id, val_list in tqdm(questions.items()):
    err_cnt = 0
    try:
        db = extractedDB[db_id]
    except KeyError:
        continue

    '''
    prompt_db 是类似这样 Flights(id, source_airport, destination_airport, depart_time)
    '''
    prompt_db = ""
    for table, attr_list in db.items():
        prompt_db += f"{table}({', '.join(attr_list)})\n"
    prompt_db += "\n"



    for val in val_list:
        question = val["question"]
        prompt_question = "Create a SQL request to "
        if question.strip().endswith("?"):
            prompt_question += "answer "
        prompt_question += question


        prompt = prompt_head + prompt_db + prompt_question
        # if random.randint(1, 100) < 3:
        #     print(prompt)
        #     print('****************************************************************************')

        prompt_rest = prompt_db + prompt_question
        continuation = gpt3.predict_unconstrained(prompt_rest, max_tokens=320, stop=[';'])

        with open(f'sql/{db_id}.sql', 'a') as f:
            p = r'/*' + prompt + r'*/'
            f.write(p + '\n' + continuation + '\n\n')

        # print(continuation, end='\n\n\n')
        try:
            sqlglot.transpile(continuation)
        except sqlglot.errors.ParseError:
            err_cnt += 1
    total = len(val_list)
    logging.error(f'{db_id}:{total-err_cnt} / {total}')