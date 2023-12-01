#!/usr/bin/env python3
"""Given a data file with student questions, get Claude results.

The questions are prompted in the exact order that they're given.

```
python -u ./claude.py \
    --access-key "" \
    --input-path questions.jsonl \
    --output-path claude_responses.jsonl.gz
```
"""
import argparse
import json
import logging
import sys
from copy import deepcopy
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from tqdm import tqdm
from xopen import xopen

logger = logging.getLogger(__name__)

def main(
    access_key,
    input_path,
    output_path,
):
    examples = []
    prompts = []

    with open("./prompt.prompt") as f:
        prompt_template = f.read().rstrip("\n")

    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            question = input_example["question"]
            
            prompt = prompt_template.format(question=question)
            prompts.append(prompt)
            examples.append(deepcopy(input_example))

    responses = []
    anthropic = Anthropic(api_key=access_key)
    for prompt in tqdm(prompts):
        try:
            completion = anthropic.completions.create(
                model="claude-2.0",
                max_tokens_to_sample=1000,
                temperature=0,
                prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
            )
            responses.append(completion.completion)
        except:
            print("Failed To Get the Response.")
        
    with xopen(output_path, "w") as f:
        for example, prompt, response in zip(
            examples, prompts, responses
        ):
            output_example = deepcopy(example)
            output_example["model_prompt"] = prompt
            output_example["model_answer"] = response
            f.write(json.dumps(output_example) + "\n")

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-key", help="API key.", required=True)
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.access_key,
        args.input_path,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
