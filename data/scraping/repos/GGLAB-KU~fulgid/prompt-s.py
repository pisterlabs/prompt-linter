import json
import os
import pathlib
import re
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

ENGINE = "gpt-3.5-turbo"
TEMPERATURE = 0.0

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

user1_msg = """Given the sentence "Sarah was a much better surgeon than Maria so ___ always got the easier cases.", which of the following options best fills in the blank?
1) Sarah
2) Maria
"""
assistant1_msg = '2'

current_dir = pathlib.Path(__file__).parent.resolve()
dataset_path = Path(os.path.join(current_dir, "dataset/dev.jsonl"))
prediction_path = Path(os.path.join(current_dir, "results/simple-predictions.json"))


def generate_prompt(data_dict):
    sentence_with_blank = data_dict['sentence'].replace("_", "___")

    options = []
    for key, value in data_dict.items():
        if 'option' in key:
            options.append(value)

    prompt = f'Given the sentence "{sentence_with_blank}", which of the following options best fills in the blank?\n'
    for i, option in enumerate(options, 1):
        prompt += f'{i}) {option}\n'

    return prompt


def load_predictions():
    if prediction_path.exists():
        with open(prediction_path, 'r') as f:
            return json.load(f)
    return {}


def update_predictions(output, data):
    predictions = load_predictions()
    qID = data['qID']

    # Use regex to extract the number
    match = re.search(r'(\d+)', output)
    if match:
        index = match.group(1)
        predictions[qID] = index
    else:
        print(f"Warning: No number found in output for qID {qID}")

    with prediction_path.open('w') as f:
        json.dump(predictions, f, indent=4)


def is_predicted_before(qID):
    predictions = load_predictions()
    return qID in predictions


def process_dataset():
    dataset_items = list(open(dataset_path, 'r'))
    for str_item in dataset_items[10:110]:
        data = json.loads(str_item)
        qID = data['qID']
        if not is_predicted_before(qID):
            prompt = generate_prompt(data)
            response = openai.ChatCompletion.create(
                model=ENGINE,
                messages=[
                    {"role": "user", "content": user1_msg},
                    {"role": "assistant", "content": assistant1_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
            )
            output = response['choices'][0]['message']['content']

            update_predictions(output, data)
            print(qID, "finished")


process_dataset()
