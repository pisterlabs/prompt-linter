"""Prompt samples using zero-shot, few-shot, and chain-of-thought (CoT) prompting methods to find attribute-values

Input
    - samples csv file
        - path = mica-character-attribute-extraction/prompt-results/samples.csv
        - contains attribute-type, id, imdb id, passage id, passage, character name, genres, answer probability fields

Output
    - completions json file
        path = mica-character-attribute-extraction/prompt-results/{zero/few/cot}.json
    - completions txt file
        path = mica-character-attribute-extraction/prompt-results/{zero/few/cot}.txt

Parameters
    - prompt method
"""

import os
import re
import json
import tqdm
import openai
import tenacity
import tiktoken
import collections
import pandas as pd

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_bool("prompt", default=False, help="set to prompt, otherwise only the expected charge is calculated")
flags.DEFINE_enum("prompt_type", default="zero", help="prompting method", enum_values=["zero", "few", "cot"])

@tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=60), stop=tenacity.stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def prompt_sample(prompt, max_tokens=256):
    try:
        response = completion_with_backoff(
            model="text-davinci-003", 
            prompt=prompt,
            temperature=0,
            max_tokens=max_tokens,
            logprobs=1
            )
        return response.to_dict()
    except Exception:
        return

def zero_shot_annot(_):
    # openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = "org-xPjDKPQ58le6x8A7CE13e8O6"
    encoding = tiktoken.encoding_for_model("text-davinci-003")

    # read samples
    data_dir = os.path.join(os.getenv("DATA_DIR"), "mica-character-attribute-extraction")
    samples_file = os.path.join(data_dir, "prompt-results/samples.csv")
    output_json_file = os.path.join(data_dir, f"prompt-results/{FLAGS.prompt_type}.json")
    output_txt_file = os.path.join(data_dir, f"prompt-results/{FLAGS.prompt_type}.txt")
    df = pd.read_csv(samples_file, index_col=None)
    df["is_goal"] = df["attr"] == "goal"
    df.sort_values("is_goal", ascending=False, inplace=True)

    # read brief goal cot prompt
    with open(os.path.join(data_dir, "attr_instr/goal/cot_prompt_brief.txt")) as fr:
        brief_goal_instr = fr.read().strip()

    # read instructions
    attrs = sorted(df["attr"].unique())
    instrs = []
    max_response_sizes = []
    avg_response_sizes = []
    for attr in attrs:
        if FLAGS.prompt_type == "cot":
            prompt_file = os.path.join(data_dir, "attr_instr", attr, "cot_prompt.txt")
        else:
            prompt_file = os.path.join(data_dir, "attr_instr", attr, "prompt.txt")
        with open(prompt_file) as fr:
            lines = fr.read().strip().split("\n")
        instr = lines[0].strip() if FLAGS.prompt_type == "zero" else "\n".join(lines).strip()
        instrs.append(instr)
        response_sizes = []
        if attr == "goal":
            i = 2
            while i < len(lines):
                while lines[i].strip() != "":
                    i += 1
                response = lines[i + 2].lstrip("Answer:").strip()
                if "CANNOT ANSWER" not in response:
                    response_sizes.append(len(encoding.encode(response)))
                i += 5
        else:
            i = 4
            while i < len(lines):
                response = lines[i].lstrip("Answer:").strip()
                if "CANNOT ANSWER" not in response:
                    response_sizes.append(len(encoding.encode(response)))
                i += 4
        max_response_sizes.append(max(response_sizes))
        avg_response_sizes.append(sum(response_sizes)/len(response_sizes))
    
    # print average response sizes
    print("response sizes =>")
    for attr, av, mx in zip(attrs, avg_response_sizes, max_response_sizes):
        print(f"\t{attr:30s} : avg = {av:.1f} tokens, max = {mx:3d} tokens")
    print()

    # prompt
    n_tokens = 0
    responses_json, responses_txt = [], []
    max_prompt_sizes = collections.defaultdict(int)
    n_times_brief_goal_prompt_used = 0
    n_times_output_tokens_lt_256 = 0
    tbar = tqdm.tqdm(df.iterrows(), total=len(df), unit="sample")
    for _, row in tbar:
        attr = row["attr"]
        tbar.set_description(attr)
        i = attrs.index(attr)
        instr = instrs[i]
        text, character = row["text"], row["character"]
        if attr == "goal":
            prompt = f"{instr}\n\nPassage:\n{text}\n\nCharacter: {character}\nAnswer:"
            if len(encoding.encode(prompt)) + 256 > 4096:
                prompt = f"{brief_goal_instr}\n\nPassage:\n{text}\n\nCharacter: {character}\nAnswer:"
                n_times_brief_goal_prompt_used += 1
        else:
            prompt = f"{instr}\n\nPassage: {text}\nCharacter: {character}\nAnswer:"
        n_prompt_tokens = len(encoding.encode(prompt))
        n_sample_tokens = 0 # sample = prompt + completion
        if FLAGS.prompt:
            max_tokens = min(256, 4096 - n_prompt_tokens)
            response = prompt_sample(prompt, max_tokens=max_tokens)
            if max_tokens < 256:
                n_times_output_tokens_lt_256 += 1
            if response is not None:
                responses_json.append(response)
                answer = re.sub(r"\s+", " ", response["choices"][0]["text"]).strip()
                responses_txt.append(answer)
                n_sample_tokens = response["usage"]["total_tokens"]
            else:
                responses_json.append({})
                responses_txt.append("ERROR")
                n_sample_tokens = n_prompt_tokens
        else:
            n_sample_tokens = n_prompt_tokens + 256
        max_prompt_sizes[attr] = max(max_prompt_sizes[attr], n_sample_tokens)
        n_tokens += n_sample_tokens
    charge = (0.02 * n_tokens) / 1000
    max_prompt_sizes = sorted(max_prompt_sizes.items())

    print(f"{n_times_brief_goal_prompt_used} times the brief goal prompt used")
    print(f"{n_times_output_tokens_lt_256} times the number of completion tokens decreased from 256\n")

    # print max tokens used per sample per attribute
    print("max tokens per sample =>")
    for attr, mt in max_prompt_sizes:
        print(f"\t{attr:30s} = {mt:4d} tokens")
    print()

    if FLAGS.prompt:
        with open(output_json_file, "w") as fw:
            json.dump(responses_json, fw, indent=2)
        with open(output_txt_file, "w") as fw:
            fw.write("\n".join(responses_txt))
        print(f"Total tokens = {n_tokens:.1f}")
        print(f"Total charge = ${charge:.1f}")
    else:
        print(f"Maximum total tokens = {n_tokens:.1f}")
        print(f"Maximum total charge = ${charge:.1f}")

if __name__=="__main__":
    app.run(zero_shot_annot)