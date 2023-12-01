"""
This file isn't used in our final version.
"""
import sys
import fire
import json
import logging
import regex as re
import random
sys.path.append("..")
from collections import Counter, defaultdict
from string import Template
from xgptscore.openai_utils import openai_completions, _chatml_to_prompt
logging.basicConfig(level=logging.WARNING)

rank_template = """
4 different models are asked to follow a given instruction to generate an answer based on a given source input.
The instruction is: ${instruction}
The source input is: ${source}
The generated output of model 1 is: ${model1_generated}
The generated output of model 2 is: ${model2_generated}
The generated output of model 3 is: ${model3_generated}
The generated output of model 4 is: ${model4_generated}
The reference output is: ${reference}

Now Please rank the 4 model's outputs from best to worst.
Please first output the rank results in the following format:
[best] [second best] [third best] [worst] (e.g. 1 2 3 4)
Then give your brief comments on why you rank the outputs in this way.
"""


def get_rank_prompts(
    item: dict
):
    random.shuffle(item['candidates'])
    rank_prompt = Template(rank_template).substitute(
        instruction=item['instruction'],
        source=item['input'],
        model1_generated=item['candidates'][0]['text'],
        model2_generated=item['candidates'][1]['text'],
        model3_generated=item['candidates'][2]['text'],
        model4_generated=item['candidates'][3]['text'],
        reference=item.get('output') or item.get("refs")[0],
    )
    return rank_prompt


def main(
    input_file: str,
    output_file: str,
    seed: int = 42,
    model_name: str = "ChatGPT",
):
    random.seed(seed)
    with open(input_file, "r") as f:
        data = json.load(f)

    rank_prompts = list(map(get_rank_prompts, data))
    chatmls = [[{"role": "system", "content": "You are an helpful AI assistant to help user find information."},
                {"role": "user", "content": prompt}] for prompt in rank_prompts]
    chatml_prompts = [_chatml_to_prompt(chatml) for chatml in chatmls]

    decoding_kwargs = {
        # "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1.0,
        "timeout": 30,
        "request_timeout": 30
    }
    results = openai_completions(
        chatml_prompts, model_name=model_name, **decoding_kwargs)
    logging.warning("Total price: {:.4f}$".format(
        sum(results['price_per_example'])))
    completions = results['completions']

    best_model_idxs = []
    model_ranks = defaultdict(list)
    for i, item in enumerate(data):
        item['rank_prompt'] = rank_prompts[i]
        item['rank_response'] = completions[i]
        try:
            first_digit_idx = re.search(r"\d", item['rank_response']).start()
            item['ranks'] = re.search(
                r"(\d)[\n ](\d)[\n ](\d)[\n ](\d)", item['rank_response'])
            if not item['ranks']:
                item['ranks'] = re.search(
                    "\[best\] (\d) \[second best\] (\d) \[third best\] (\d) \[worst\] (\d)", item['rank_response'])
            if not item['ranks']:
                item['ranks'] = re.search(
                    "\[best\] Model (\d)[\n ]\[second best\] Model (\d)[\n ]\[third best\] Model (\d)[\n ]\[worst\] Model (\d)", item['rank_response'])
            # item['ranks'] = item['rank_response'][first_digit_idx:item['rank_response'].index("\n")].split(" ")
            item['ranks'] = [int(rank) for rank in item['ranks'].groups()]
        except Exception:
            print(item['ranks'])
        for j, cand in enumerate(item['candidates']):
            cand['scores']['gpt_rank_{}'.format(
                model_name)] = - item['ranks'][j]
            model_ranks[cand['source']].append(item['ranks'][j])
        best_model_idxs.append(item['ranks'][0])

    print(Counter(best_model_idxs))
    for model, ranks in model_ranks.items():
        c = Counter(ranks)
        print(model, sorted(c.items(), key=lambda x: x[0]))
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logging.warning(f"Saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
