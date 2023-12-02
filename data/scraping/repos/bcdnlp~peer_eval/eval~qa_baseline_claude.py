import json

import shortuuid
from tqdm import tqdm

import anthropic_api

MODEL_ID = 'claude:20230617'

with open('data/vicuna/vicuna_questions.jsonl', 'r') as f:
    data = [json.loads(line) for line in f.readlines()]

answers = []
for d in tqdm(data):
    idx = d['question_id']
    question = d['text']
    category = d['category']

    answer = anthropic_api.call(['You are a helpful assistant.', question])

    answers.append({"answer_id": shortuuid.uuid(),
                    "question_id": idx,
                    "model_id": MODEL_ID,
                    "text": answer})

with open('answer_claude.jsonl', 'w') as f:
    for answer in answers:
        f.write(json.dumps(answer) + '\n')

