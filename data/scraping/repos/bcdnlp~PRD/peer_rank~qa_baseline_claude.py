import json

import anthropic
import shortuuid
from tqdm import tqdm

MODEL_ID = 'claude:20230617'

with open('../keys/anthropic_key', 'r') as f:
    api_key = f.read().strip()
client = anthropic.Client(api_key)

with open('../data/vicuna80/vicuna_questions.jsonl', 'r') as f:
    data = [json.loads(line) for line in f.readlines()]

def prompt_formatting(history):
    prompt = f"{anthropic.HUMAN_PROMPT} {history[0]}"
    for idx, content in enumerate(history[1:]):
        if 0 == idx % 2:
            message = f"{anthropic.HUMAN_PROMPT}\n{content}"
        else:
            message = f"{anthropic.AI_PROMPT}\n{content}"
        prompt = ''.join([prompt, message])

    prompt = ''.join([prompt, f"{anthropic.AI_PROMPT}"])

    return prompt

def call(history):
    prompt = prompt_formatting(history)
    while True:
        try:
            response = client.completion(
                           prompt=prompt,
                           stop_sequences=[anthropic.HUMAN_PROMPT],
                           max_tokens_to_sample=500,
                           model='claude-1',
                           temperature=0.2,
                           )
            break
        except Exception as e:
            time.sleep(2)
            print('Errrrrrrrrrrrrrrrrrr', str(e))

    prediction = response['completion']

    return prediction
 
answers = []
for d in tqdm(data):
    idx = d['question_id']
    question = d['text']
    category = d['category']

    answer = call(['You are a helpful assistant.', question])

    answers.append({"answer_id": shortuuid.uuid(),
                    "question_id": idx,
                    "model_id": MODEL_ID,
                    "text": answer})

with open('../data/vicuna80/generations/answer_claude.jsonl', 'w') as f:
    for answer in answers:
        f.write(json.dumps(answer) + '\n')

