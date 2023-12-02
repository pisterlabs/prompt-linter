import os
import sys
import pdb
import json
import math
import openai
import argparse
from tqdm import tqdm
from typing import List
sys.path.append('./GPT-3/optimization')
sys.path.append('./GPT-3')

from utils.gpt_utils import gpt_inference, chatgpt_single_turn_inference
from Constants import NIV2_OUT_DOMAIN_TEST_TASKS_SET2, NIV2_TEST_TASK_FORMAT_INSTRUCTION
TEST_TASKS = NIV2_OUT_DOMAIN_TEST_TASKS_SET2


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    # print(f'Read {len(result)} data from {path}')
    return result


def save_list_as_jsonl(path: str, data):
    assert path.endswith('.jsonl')
    with open(path, 'w', encoding='utf8') as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write('\n')
    # print(f'Saved {len(data)} data to {path}')


def read_txt(path: str):
    with open(path, 'r', encoding='utf8') as fin:
        text = fin.read().strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, default='data/niv2_english')
    parser.add_argument('-naive_instruction_file', type=str, default='naive_instruction.txt')
    parser.add_argument('-meta_prompt_dir', type=str, default='GPT-3/optimization/instruction_generation_templates/zero-shot')
    parser.add_argument('-output_file_name', type=str, default='zs_text003_prompt_19.json')

    # GPT-3 API arguments
    parser.add_argument('-model', type=str, default='text-davinci-003',
                        help='Which GPT-3 model to use')
    parser.add_argument('-batch_size', type=int, default=20,
                        help='Batch size of GPT-3 API calls')
    parser.add_argument('-max_tokens', type=int, default=300,
                        help='Max number of output tokens of GPT-3')
    parser.add_argument('-temperature', type=float, default=1.0,
                        help='Temperature parameter of GPT-3. 0: greedy decoding. 1: sampling based on probability')
    parser.add_argument('-top_p', type=float, default=1.0,
                        help='Sampling probability threshold in nucleus sampling')
    parser.add_argument('-best_of', type=int, default=5,
                        help='Maximum number of sampled outputs')
    parser.add_argument('-num_return', type=int, default=3,
                        help='How many output sequences to return. Note that this argument also applies to ChatGPT '
                             'if using ChatGPT to generate multiple instructions')
    parser.add_argument('-logprobs', type=int, default=None,
                        help='The number of tokens to return probability')
    parser.add_argument('-timeout', type=int, default=40,
                        help='Wait time for the response from OpenAI API')
    parser.add_argument('-sleep', type=int, default=1,
                        help='How long to sleep if rate limit is reached')
    parser.add_argument('-api_key', type=str, required=True, help='OPENAI API key')
    args = parser.parse_args()

    meta_prompt_files = [
        "zs_niv2_demo_task_1.txt",
        "zs_niv2_demo_task_3.txt",
        "zs_niv2_demo_task_5.txt",
        "zs_one-paragraph_instruction.txt",
        "zs_one-sentence_instruction.txt",
        "zs_step-by-step_instruction.txt"
    ]
    meta_prompt_list = [read_txt(os.path.join(args.meta_prompt_dir, f)) for f in meta_prompt_files]

    for i, task in tqdm(enumerate(TEST_TASKS)):
        task_dir = os.path.join(args.data_dir, task)

        print(f"{'-' * 15}{task}({i + 1}/{len(TEST_TASKS)}){'-' * 15}")

        task_description = open(os.path.join(task_dir, args.naive_instruction_file), 'r', encoding='utf8').read().strip()
        if task in NIV2_TEST_TASK_FORMAT_INSTRUCTION.keys():
            task_description = task_description + " " + NIV2_TEST_TASK_FORMAT_INSTRUCTION[task]
        
        output_path = os.path.join(task_dir, args.output_file_name)
        if os.path.exists(output_path):
            print(f'Output path {output_path} already exists! Skip to next task...')
            continue

        # gather all meta prompts from all examples before putting them into batches
        all_meta_prompts = []
        for meta_prompt_template in meta_prompt_list:
            meta_prompt = meta_prompt_template.replace('{{task}}', task_description)
            all_meta_prompts.append(meta_prompt)

        if args.model == "text-davinci-003":
            
            openai.api_key = args.api_key
            instruction_list, logprob_dict = gpt_inference(
                inputs_with_prompts=all_meta_prompts,
                model=args.model,
                max_tokens=args.max_tokens,
                num_return=args.num_return,
                best_of=args.best_of,
                temperature=args.temperature,
                top_p=args.top_p,
                logprobs=args.logprobs,
                stop=None,
                timeout=args.timeout,
                sleep=args.sleep
            )

        elif args.model in ['gpt-3.5-turbo', 'gpt-4-0314', 'gpt-4']:
            openai.api_key = args.api_key
            instruction_list = []

            for meta_prompt in all_meta_prompts:
                instructions = chatgpt_single_turn_inference(
                    inputs_with_prompts=meta_prompt,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    num_return=args.num_return,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=None,
                    timeout=args.timeout,
                    sleep=args.sleep
                )
                instruction_list.extend(instructions)

        else:
            raise ValueError(f"Model {args.model} is not supported!")

        assert len(instruction_list) == len(all_meta_prompts) * args.num_return

        instruction_list = [task_description] + instruction_list

        json.dump(instruction_list, open(output_path, 'w', encoding='utf8'), indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()

