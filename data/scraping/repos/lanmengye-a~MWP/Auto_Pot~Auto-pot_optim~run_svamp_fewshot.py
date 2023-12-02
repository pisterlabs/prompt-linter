import openai
from time import sleep
from tool import *
from typing import Dict, Any
from datetime import datetime
from tqdm import tqdm
import os
import json
import argparse
from collections import Counter
import collections



def create_reader_request(example: Dict[str, Any]) -> str:
    if not example["Body"].endswith('.'):
        example["Body"] += '.'
    string = f'{example["Body"]} {example["Question"]}'
    return string



def get_rationales(args, question,few_prompt):
    openai.api_key = args.key
    question = question.strip().strip("# Question:")
    prefix ="""
# Read the following passages to answer questions with Python code,store the result as a 'ans' variable:
"""
    midfix = """
# Principles for solutions
# Firstly , we can define the variables , bind them with every number's semantics in Question
# Secondly, the quantity of numbers is equal with the quantity of numbers in Question
# Thirdly, Each equation only uses previously introduced variables instead of the number itself
# Finally, store the result as a 'ans' variable"""


    if few_prompt is not None:
        message = prefix+"\n"+few_prompt.strip()+"\n\n"+"# Q:"+question+"\n"+midfix+"\n"+"# T:"


    practice = 1
    while practice!=0:
        try:
            result = openai.Completion.create(
                engine='text-davinci-002',
                prompt=message,
                # api_key=os.getenv(args.key),
                max_tokens=300,
                logit_bias={"1303": -2},
                temperature=0.5,
                top_p=1,
                n=30,
                stop=['\n\n'],
                logprobs=1
            )
            codes = parse_api_result(result)
            result_counter = Counter()
            result_dict = collections.defaultdict(list)

            for idx,r in enumerate(codes):
                ans = safe_execute(r)
                ans = floatify_ans(ans)
                if ans is not None and not isinstance(ans, str):
                    result_counter.update([(ans)])
                    result_dict[ans].append(idx)
                    # code_valid.append(r)
            if len(result_counter) > 0:
                prediction = result_counter.most_common(1)[0][0]
            else:
                prediction = None
            if prediction is not None:
                program = codes[result_dict[prediction][0]]
                break
            else:
                program = None
                practice -= 1
        except Exception as e:
            print(e)
            sleep(3)
    print("prediction:", prediction)
    if program is not None:
        print("program:", program)
    return prediction, program






if __name__ == "__main__":
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default='OPENAI_KEY', type=str)
    parser.add_argument("--dry_run", default=False, action='store_true')
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    args = parser.parse_args()
#     with open('dataset/SVAMP/SVAMP.json') as f:
#         svamp_data = json.load(f)
#     correct, wrong = 0, 0
#     openai.api_key = "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"
#     svamp_data = svamp_data[args.start:args.end]
#     filename = f'outputs/svamp_zs_s{args.start}_e{args.end}_{dt_string}.jsonl'
#     print(filename)
#     writer = open(filename, 'w')
#     for example in tqdm(svamp_data):
#         full_prompt = f"""
# import math
# import numpy as np
#
# {create_reader_request(example)}
# # Answer this question by implementing a solver() function.
# def solver():
#     # Let's think step by step to derive the answer, and then return the answer
#     # According to the question, we can define the variables:
# """
#         if args.dry_run:
#             print(full_prompt)
#             print('=======================')
#             continue
#
#         # greedy decoding
#         got_result = False
#         while not got_result:
#             try:
#                 result = openai.Completion.create(
#                     engine='text-davinci-002',
#                     prompt=full_prompt,
#                     api_key=os.getenv(args.key),
#                     max_tokens=300,
#                     logit_bias={"1303": -2},
#                     temperature=0.5,
#                     top_p=1,
#                     n=30,
#                     stop=['\n\n'],
#                     logprobs=1
#                 )
#                 got_result = True
#             except Exception:
#                 sleep(3)
#         result_counter = Counter()
#         codes = parse_api_result(result)
#         for r in codes:
#             r = synthesize_program(r,full_prompt)
#             ans = safe_execute(r)
#             ans = floatify_ans(ans)
#             if ans is not None:
#                 result_counter.update([abs(ans)])
#
#         if len(result_counter) > 0:
#             prediction = result_counter.most_common(1)[0][0]
#         else:
#             prediction = None
#         result =  full_prompt+result['choices'][0]['text']
#         program = synthesize_program(result,full_prompt)
#         ans = safe_execute(program)
#         prediction = floatify_ans(ans)
#
#         gt_ans = float(example['Answer'])
#         if finqa_equal(prediction, gt_ans):
#             correct += 1
#         else:
#             wrong += 1
#
#         print(program)
#         print(prediction, '$', gt_ans, '$', correct / (correct + wrong))
#
#         tmp = {'question': example['Question'], 'passage': example['Body'],
#                'executed': prediction, 'generated': program, 'answer': gt_ans}
#         writer.write(json.dumps(tmp) + '\n')
#
#     writer.close()
#     print()
#     print(correct / (correct + wrong))
