import argparse
import json
import os
import time

import openai
from tqdm import tqdm

import shortuuid
import asyncio
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
openai.api_key = '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'
async def dispatch_openai_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]


def gen_prompt(reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2):
    # Default to general category (index=0)
    reviewer_idx = 0
    for idx, reviewer in enumerate(reviewer_jsons):
        if reviewer["category"] == cat:
            reviewer_idx = idx
            break
    prompt_id = reviewer_jsons[reviewer_idx]["prompt_id"]
    prompt_json = prompt_jsons[prompt_id - 1]
    assert prompt_json["prompt_id"] == prompt_id

    sys_prompt = prompt_json["system_prompt"]
    prompt_template = prompt_json["prompt_template"]
    defaults = prompt_json["defaults"]
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, **defaults
    )

    return sys_prompt, prompt, reviewer_idx + 1


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-qa", "--qa-file")
    parser.add_argument("-k1", "--key-1")
    parser.add_argument("-k2", "--key-2")
    parser.add_argument("-p", "--prompt-file")
    parser.add_argument("-r", "--reviewer-file")
    parser.add_argument("-o", "--output-review-file")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()
    reviewer_jsons = get_json_list(args.reviewer_file)
    prompt_jsons = get_json_list(args.prompt_file)
    qa_jsons = json.load(open(args.qa_file))

    message_list = []
    review_jsons = []
    total_len = len(qa_jsons)
    question_idx_list = list(range(total_len))

    for i in question_idx_list:
        ques = qa_jsons[i]["text"]
        cat = qa_jsons[i]["category"]
        ans1 = qa_jsons[i][args.key_1]
        ans2 = qa_jsons[i][args.key_2]
        sys_prompt, prompt, reviewer_id = gen_prompt(
            reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2
        )
        review_id = shortuuid.uuid()
        review_jsons.append(
            {
                "review_id": review_id,
                "question_id": qa_jsons[i]["question_id"],
                "answer1": ans1,
                "answer2": ans2,
                "reviewer_id": reviewer_id,
                "metadata": {},
            }
        )
        message =[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                    },
        ]
        message_list.append(message)
    predictions = []
    for i in tqdm(range(0,len(message_list),10)):
        batch_predictions = asyncio.run(
            dispatch_openai_requests(
                messages_list=message_list[i:i+10],
                model="gpt-3.5-turbo",
                temperature=0.2,
                max_tokens=args.max_tokens,
                top_p=1.0,
            )
        )
        predictions += batch_predictions
    with open(f"{args.output_review_file}", "w") as output_review_file:
        for idx, prediction in enumerate(predictions):
            review = prediction['choices'][0]['message']['content']
            scores = parse_score(review)
            review_jsons[idx]["text"] = review
            review_jsons[idx]["score"] = scores
            output_review_file.write(json.dumps(review_jsons[idx]) + "\n")
