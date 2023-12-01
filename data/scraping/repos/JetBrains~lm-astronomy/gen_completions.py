from enum import Enum
import argparse
import openai
import json
import time
import os
from utils import Dataset, update_json_file, create_indices, Entity

openai.api_key = os.getenv('openai_api_key')
with open(f"./data/prompts.json", 'r') as f:
    prompts = json.load(f)


class Model(str, Enum):
    chat_gpt = 'gpt-3.5-turbo-16k'
    gpt_4 = "gpt-4-0613"


def try_request(engine: str, cur_prompt: str, use_function: bool, retries: int = 0):
    if retries >= 5:
        return
    try:
        prompt_chat = [
            {"role": "system",
             "content": "You are a helpful assistant that extracts structured information from ATel and GCN Circulars datasets. Don't make assumptions about what values to plug into functions, but use only information from text."},
            {"role": "user",
             "content": cur_prompt}
        ]
        if use_function:
            response = openai.ChatCompletion.create(
                model=engine, functions=prompts["function"], messages=prompt_chat,
                function_call={"name": "extract_entities"})
            response = json.loads(response.choices[0]['message']['function_call']['arguments'])
        else:
            prompt_chat = [
                {"role": "system", "content": "You are a helpful assistant, who knows about astronomy."},
                {"role": "user", "content": cur_prompt}
            ]
            response = openai.ChatCompletion.create(
                model=engine, messages=prompt_chat)
            response = eval(response['choices'][0]['message']['content'])
        return response
    except openai.error.InvalidRequestError:
        return
    except Exception as e:
        print(f'Retrying after catching an exception, try {retries + 1}\n{e}')
        time.sleep(90)
        return try_request(engine, cur_prompt, use_function, retries + 1)


def try_prompt(dataset, engine, indices, shots_num, dataset_name, entity_name):
    json_output = {}
    output_filename = f"./data/{dataset_name}/{entity_name}/grouped_completions.json" if entity_name \
        else f"./data/{dataset_name}/function_completions.json"

    for i in indices:
        json_output[i] = []
        cur_prompt = prompts["prompt"][f"determine_{entity_name}"].format(prompts["group"][entity_name], dataset[i]) \
            if entity_name else dataset[i]
        for j in range(shots_num):
            response = None
            while not response and len(dataset[i]) > 0:
                response = try_request(engine, cur_prompt, entity_name is None)
            json_output[i].append(response if response else [])
        update_json_file(output_filename, json_output)


def get_completions(dataset_name, engine, shots_num, indices_path=None, entity_name=None):
    if entity_name:
        with open(f'./data/{dataset_name}/{entity_name}/ranked_completions.json') as json_file:
            dataset = json.load(json_file)
    else:
        with open(f'./data/{dataset_name}/dataset.json') as json_file:
            dataset = json.load(json_file)
    indices = create_indices(dataset_name, indices_path)
    try_prompt(dataset, engine, indices, shots_num, dataset_name, entity_name)


def main():
    parser = argparse.ArgumentParser(description="Generate completions for given messages.")

    parser.add_argument("-d", "--dataset", required=True, help="Dataset name", type=Dataset, choices=list(Dataset))
    parser.add_argument("-e", "--engine", required=True, help="OpenAI model compatible with ChatCompletion API",
                        type=Model, choices=list(Model))
    parser.add_argument("-s", "--size", type=int, help="Sample size", default=3)
    parser.add_argument("-i", "--indices_path", type=str, help="Path to file with indices", default=None)
    parser.add_argument("-en", "--entity_name", help="Entity name", type=Entity,
                        choices=[i for i in list(Entity) if i != Entity.OBJECT_NAME], default=None)

    args = parser.parse_args()

    get_completions(args.dataset, args.engine, args.size, args.indices_path, args.entity_name)


if __name__ == "__main__":
    main()
