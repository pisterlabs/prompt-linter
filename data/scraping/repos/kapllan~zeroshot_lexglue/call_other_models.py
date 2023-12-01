import json
import os
import random
import tqdm
from handle_llama_model import get_llama_pipeline
from handle_claude_model import anthropic
import argparse
from data import DATA_DIR
from build_instructions.templates import TEMPLATES
from constants import *
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import time

random.seed(42)

LIMIT = 200


def main(args):
    if 'llama' in args.model_name:
        model = get_llama_pipeline()
    elif args.model_name == 'claude-2':
        model = anthropic

    model_specific_output_folder_name = DATA_DIR + f'/{args.model_name}_{args.zeroshot_output_path}/'
    model_specific_output_filename = os.path.join(model_specific_output_folder_name,
                                                  f'{args.dataset_name}_{args.model_name}_predictions.jsonl')
    if not os.path.exists(model_specific_output_folder_name):
        os.mkdir(model_specific_output_folder_name)
    if not os.path.exists(model_specific_output_filename):
        open(model_specific_output_filename, 'w').close()

    dataset = []

    with open(os.path.join(DATA_DIR + '/instruction-following-examples/', f'{args.dataset_name}.jsonl')) as in_file:
        for line in in_file:
            sample_data = json.loads(line)
            dataset.append(sample_data)

    predictions = []
    if os.path.exists(model_specific_output_filename):
        with open(model_specific_output_filename) as in_file:
            for line in in_file:
                predictions.append(json.loads(line))

    # dataset = dataset[:LIMIT]
    # predictions = predictions[:LIMIT]
    for idx, example in tqdm.tqdm(enumerate(dataset)):
        if idx < len(predictions) and predictions[idx]['prediction'] is not None:
            dataset[idx]['prediction'] = predictions[idx]['prediction']
            print(f'Predictions for example #{idx} is already available!')
            continue
        else:
            try:
                if 'llama' in args.model_name:
                    response = model(example['input_text'])[0]["generated_text"]
                elif args.model_name == 'claude-2':
                    response = anthropic.completions.create(
                        model="claude-2",
                        max_tokens_to_sample=MAX_OUTPUT_LENGTH,
                        prompt=f"{HUMAN_PROMPT} {example['input_text']}{AI_PROMPT}",
                    ).completion
                dataset[idx]['prediction'] = response
            except Exception as inst:
                print(inst)
                dataset[idx]['prediction'] = None
        time.sleep(1)

    with open(model_specific_output_filename, 'w') as file:
        for example in dataset:
            file.write(json.dumps(example, ensure_ascii=False) + '\n')


parser = argparse.ArgumentParser(description='Prompting GPT')
parser.add_argument("--dataset_name", type=str,
                    default='ledgar', help="Name of dataset as stored on HF")
parser.add_argument("--truncate_demonstrations", type=int,
                    default=100, help="Truncation of demonstrations")
parser.add_argument("--zeroshot_output_path", help="Define the zero shot putput directory.",
                    default="zero-shot-predictions")
parser.add_argument("--model_name", type=str, help="model name")

args = parser.parse_args()

main(args)
