import argparse
import json
import os
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from tqdm import tqdm

# import shortuuid
import asyncio
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_score(review):
    try:
        score = float(review.split('\n')[0])
    except Exception as e:
        if('score:' in review):
            score = float(review.split('score:')[1].split('\n')[0])
        elif('Score:' in review):
            score = float(review.split('Score:')[1].strip('\n')[0])
        else:           
            logger.error(
                f"{e}\nContent: {review}\n" "You must manually fix the score pair."
            )
            score = -1
    
    return score

def find_error_items(alpaca_data,alpaca_data_cleaned_archive):
    alpaca_data_cleaned_archive_str = set([str(d) for d in alpaca_data_cleaned_archive])
    dirty_list = []
    for i, x in enumerate(alpaca_data):
        x = str(x)
        if(x not in alpaca_data_cleaned_archive_str):
            dirty_list.append(i)
    return dirty_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude-based rating.")
    parser.add_argument("-o", "--output-review-file")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="the batch size to call the Claude."
    )
    args = parser.parse_args()
    message_list = []
    # alpaca_data_cleaned_archive = json.load(open("./alpaca_data_cleaned_archive.json"))
    alpaca_data = json.load(open("./alpaca_data.json"))
    system_prompt = "We would like to request your response to the instruction and the given input displayed following."


    '''
    rating according to the helpfulness
    '''
    # user_prompt = "Please rate according to the helpfulness of the response to the instruction and the input. Each assistant receives an score on a scale of 0 to 5, where a higher score indicates higher level of the helpfulness. Please first output a single line containing value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias. \n\n"
    # dirty_list = find_error_items(alpaca_data, alpaca_data_cleaned_archive)
    '''
    rating according to the accuracy
    '''
    print(f"Alpaca data pairs: {len(alpaca_data)}")
    for i in range(len(alpaca_data)):
        # import pdb; pdb.set_trace()
        triplet = "###Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n".format_map(alpaca_data[i])
        eval_prompt = system_prompt + triplet 
        message = f"{HUMAN_PROMPT}{eval_prompt}{AI_PROMPT}"
        # import pdb; pdb.set_trace()
        message_list.append(message)
    predictions = []
    i = 0
    wait_base = 10
    retry = 0
    batch_size = args.batch_size
    pbar = tqdm(total=len(message_list))
    anthropic = Anthropic()
    # Try 10 messages firstly
    while(i<10):
    # while(i<len(message_list)):
        try:
            completion = anthropic.completions.create(
            model="claude-2",
            temperature=0.0,
            max_tokens_to_sample=1024,
            prompt=message_list[i],
            )
            predictions.append(completion.completion)
            i += 1
            wait_base = 10
            pbar.update(batch_size)
        except:
            retry += 1
            print("Batch error: ",i, i+10)
            print("retry number: ", retry)
            time.sleep(wait_base)
            wait_base = wait_base*2
    pbar.close()

    outputs = []
    for idx, prediction in tqdm(enumerate(predictions)):
        review = prediction
        # score = parse_score(review)
        triplet = "###Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}\n\n".format_map(alpaca_data[idx])
        meta_output = {
                    "Instruction":alpaca_data[idx]['instruction'],
                    "Input": alpaca_data[idx]['input'],
                    "Response": prediction
                }
        outputs.append(meta_output)
    
    with open(f"{args.output_review_file}", "w") as output_review_file:
        json.dump(outputs, output_review_file, indent=4)

# if __name__=="__main__":
#     review = "3.5\n\nThe response provides a basic understanding of the difference between individual and societal performance. However, it lacks depth and does not provide specific examples or analysis to support the comparison and contrast. The language used is clear and concise, but the response could benefit from more elaboration and explanation. Overall, the response is helpful to a certain extent, but could be improved with more detail and analysis."
#     print(parse_score(review))
