import os 
import ast
import json
import openai
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict
from transformers import GPT2Tokenizer

from openai_inference import chat_inference


def parse_result(result: dict) -> list:
    """ 
    Parse ChatGPT generation from a string to list of entities

    Args:
        result: ChatGPT result

    Returns:
        list of entities
    """

    result = result['content'].strip()
    if 'None' in result:
        return []
    else:
        result = result.split('\n')[-1]
        if result[-1] == "'":
            result += ']'
        elif result[-1] == ',':
            result = result[:-1] + ']'

        try:
            result = ast.literal_eval(result)
        except:
            result = [] 
        return result



def load_json(path: str) -> List[Dict[str, str]]:
    """
    Load json file from a given path

    Args:
        path: path to the json file

    Returns:
        json file as a dictionary 
    """
    with open(path, 'r') as f:
        data = json.load(f)
    f.close()
    return data


def count_tokens(inp: str) -> int:
    """
    Count the number of tokens in a given string

    Args:
        inp: input string

    Returns:
        number of tokens
    """
    inp_tokenized = tokenizer(inp).input_ids
    return len(inp_tokenized)


def make_example(example_dict: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Make n-shot in-context example for ChatGPT

    Args:
        example_dict: dicionary of example prompts

    Returns:
        example prompt in ChatGPT format (https://platform.openai.com/docs/guides/chat/introduction)
    """
    header = example_dict[0]
    example_list = example_dict[1:]
    total_example = int(len(example_list) / 4)

    example_idx = np.random.choice(total_example, args.num_shot, replace=False)
    out = []
    total_token = 0
    for idx in example_idx:
        out.append(example_list[idx * 4: idx * 4 + 4])
        total_token += sum([count_tokens(item['content']) for item in example_list[idx: idx + 4]])

    out = [lst for sublst in out for lst in sublst]

    out = [header] + out
    total_token += count_tokens(header['content'])
    print(f'Example contains {total_token} tokens.')
    return out


def parse_result(result: dict) -> list:
    """
    Parse ChatGPT generation from a string to list of entities

    Args:
        result: the generated dictionary from ChatGPT

    Returns:
        a list of entities
    """
    res = result['content'].split('\n')[-1]
    if "'" not in res:
        return []
    else:
        # clean generation
        if res.strip()[-1] == ',':
            res = res.strip()[:-1] + ']'
        elif res.strip()[0] == '[' and res.strip()[-2:] != "']":
            res = res + "']"

        # convert generation to list
        try:
            return ast.literal_eval(res)
        except:
            return []


def get_args() -> argparse.Namespace:
    """
    Get arguments from command line

    Returns:
        arguments as a namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_shot', type=int, default=3, help='number of shots for in-context learning'
    )
    parser.add_argument(
        '--num_run', type=int, default=3, help='number of runs for each attribute'
    )
    return parser.parse_args()


def main():
    data = load_json('../../../data/data_in_new_format/dev-data-reformatted-v4.json')
    attr_example = load_json('./assets/attr_example.json')
    attr_template = load_json('./assets/attr_template.json')
    openai.api_key_path = os.path.expanduser('~/harry.key')
    np.random.seed(42)

    global args, tokenizer
    args = get_args()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    attr_example = make_example(attr_example)

    for proc_id, proc_content in tqdm(data.items(), position=0, leave=False):
        cur_goal = proc_content['goal']
        cur_steps = proc_content['steps']
        cur_clusters = proc_content['clusters']
        
        cur_step_narratives = []
        for i, step in enumerate(cur_steps):
            if i == 0:
                cur_step_narratives.append(f'First, I {step.lower()}')
            else:
                cur_step_narratives.append(cur_step_narratives[i-1] + f' Then, I {step.lower()}')

        cur_goal = f'I am trying to {cur_goal.strip().lower()}. '
        cur_steps = cur_goal + cur_step_narratives[-1]

        # over generate the cluster entries
        for entity_id, entity_content in tqdm(cur_clusters.items(), position=1, leave=False):
            cur_attr_cluster = entity_content['attribute_cluster']

            # generate the attribute clusters
            for attr_id, attr_content in cur_attr_cluster.items():
                cur_template = deepcopy(attr_template)
                # add procedure context
                cur_context = cur_template[0]['content'].replace('{context}', cur_steps) + ' Do you get the procedure?'
                cur_template[0]['content'] = cur_context

                # add entity context
                cur_attr_context = f'Here are some attributes that describe the same property of an object: {attr_content} of {entity_id}. What are some alternative names for this property? Organize them in a list. Answer \"None\" if there is no alternative name.'
                cur_template[2]['content'] = cur_template[2]['content'].replace('{attr_context}', cur_attr_context)

                # remove answer from the template
                cur_input = attr_example + cur_template[:-1]

                all_output = []
                for _ in range(args.num_run):
                    cur_output = chat_inference(cur_input)
                    cur_output = parse_result(cur_output)
                    all_output.extend(cur_output)

                data[proc_id]['clusters'][entity_id]['attribute_cluster'][attr_id].extend(list(set(all_output)))

    with open('../../../data/data_in_new_format/dev-data-reformatted-v4-attribute-overgenerated.json', 'w') as f:
        json.dump(data, f, indent=4)
    f.close()


if __name__ == "__main__":
    main()

