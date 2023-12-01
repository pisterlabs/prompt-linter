"""
Copyright 2023 Björn Plüster

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from datasets import load_dataset
from tqdm import tqdm, trange
import guidance
import json
from pathlib import Path
from datasets.utils.logging import disable_progress_bar
#disable_progress_bar()
                                                      
# set the default language model used to execute guidance programs
guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo-0301", max_calls_per_min=5000, api_key="./openai_key.txt") 

structure_program = guidance(
'''
{{#system~}}
You are a helpful assistant that translates json from English to German.
{{~/system}}

{{#user~}}
Translate the following json to german. 
It consists of a question and four possible answer choices.
Be as precise as possible. Keep the exact json format and do not translate the keys.

{{input}}
{{~/user}}

{{#assistant~}}
{{gen 'output' temperature=0.5 top_p=1}}
{{~/assistant}}
''', stream=False)

labels = ["A", "B", "C", "D"]

def translate_example(example, depth=0):
    ex = {
        "question": example["question"],
        "choices": example["choices"]["text"]
    }

    try:
        json_input = json.dumps(ex)
        if depth > 0:
            out = structure_program(
                input=json_input,
                cache_seed=depth
            )
        else:
            out = structure_program(
                input=json_input
            )
    except Exception as e:
        example["question_de"] = ""
        example["choices_de"] = {"text": ["", "", "", ""], "label": labels}
        example["translation_de"] = ""
        return example
    try:
        translated = json.loads(out["output"])
        example["question_de"] = translated["question"]
        example["choices_de"] = {"text": translated["choices"], "label": labels}
        example["translation_de"] = out.get("output", "")
    except Exception as e:
        if depth < 5:
            return translate_example(example, depth=depth+1)
        example["question_de"] = ""
        example["choices_de"] = {"text": ["", "", "", ""], "label": labels}
        example["translation_de"] = out.get("output", "")
    return example


dataset = load_dataset("ai2_arc", "ARC-Challenge", split={"test": "test", "validation": "validation"})

output_dir = Path("outputs_arc_challenge_de")
output_dir.mkdir(exist_ok=True)
num_shards = 5
for split in ["test", "validation"]:
    ds = dataset[split]
    for i in trange(num_shards, desc=f"Translating {split} shards"):
        shard = ds.shard(num_shards=num_shards, index=i)
        shard = shard.map(translate_example, num_proc=16)
        shard.to_json(output_dir / f"{split}-{i:03d}.json")

# Combine shards

json_files = {
    "test": [str(x) for x in output_dir.glob(f"test-*.json")],
    "validation": [str(x) for x in output_dir.glob(f"validation-*.json")]
}
dataset = load_dataset("json", data_files=json_files)
dataset.push_to_hub("bjoernp/arc_challenge_de")

for split in ["test", "validation"]:
    ds = dataset[split]
    # count examples with empty translation
    empty = ds.filter(lambda x: x["translation_de"] == "")
    print(f"Empty translations in {split}: {len(empty)}")
    # count examples with question translation
    empty = ds.filter(lambda x: x["question_de"] == "")
    print(f"Empty question translations in {split}: {len(empty)}")

