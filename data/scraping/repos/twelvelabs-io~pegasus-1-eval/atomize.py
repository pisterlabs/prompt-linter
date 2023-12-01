
import os
import json
from tqdm import tqdm
# import timeout_decorator
import ast
import openai

# put your openai api key here
your_api_key = ''
openai.api_key = your_api_key

def call_gpt(system_prompt, user_prompt, gpttype):
    completion = openai.ChatCompletion.create(
        model=gpttype, # gpt-3.5-turbo
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.01
    )
    result = completion["choices"][0]["message"]["content"]
    return result


def atomize(sent, gpttype):

    system_prompt = "You are excellent natural language assistant. I want output a list that can be loaded with json.loads. "
    user_prompt = f"You will be given a natural language passage. I want you to break them into a list of atomic sentences. Only give me json.loads-able list. \n The passage: {sent}\n list of atomic sentences"
    result4 = call_gpt(system_prompt, user_prompt,gpttype)
    return result4


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_fp', type=str, default='/home/ubuntu/ray/gpt-based-evaluation/vgpt_ours_500_2.json')
    parser.add_argument('--save_fp', type=str, default='/home/ubuntu/ray/gpt-based-evaluation/vgpt_ours_500_2_atomized.json')
    parser.add_argument('--gpttype', type=str, default='gpt-3.5-turbo')
    args = parser.parse_args()

    with open(args.src_fp, "r") as f:
        pred_dict = json.load(f)

    save_fp = args.save_fp

    if os.path.exists(save_fp):
        with open(save_fp, "r") as f:
            results = json.load(f)
    else:
        results = {}

    save_cnt = 0
    for vkey in pred_dict.keys():
        print(f'vkey: {vkey}')
        if vkey in results:
            continue
        pred = pred_dict[vkey]
        pred_atomic = atomize(pred, args.gpttype)
        results[vkey] = {'pred_atomic': pred_atomic,  'pred': pred}
        
        if save_cnt % 10 == 0:
            with open(save_fp, 'w') as fp:
                json.dump(results, fp, indent=4)
        save_cnt += 1

    with open(save_fp, 'w') as fp:
        json.dump(results, fp, indent=4)