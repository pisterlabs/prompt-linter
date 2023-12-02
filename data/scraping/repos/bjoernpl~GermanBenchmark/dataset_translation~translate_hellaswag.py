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
import random
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
It consists of a context and four possible continuations. Make sure that the translation of each continuation is coherent with the context.
Be as precise as possible. Keep the exact json format and do not translate the keys.

{{input}}
{{~/user}}

{{#assistant~}}
{{gen 'output' temperature=0.5 top_p=1}}
{{~/assistant}}
''', stream=False)

def fix1(example):
    translation = example["translation_de"] + "}"
    try:
        json.loads(translation)
        return translation
    except Exception as e:
        raise e

def fix2(example):
    translation = example["translation_de"].replace('"endings":', '"endings": [')
    try:
        json.loads(translation)
        return translation
    except Exception as e:
        raise e

def fix3(example):
    translation = example["translation_de"]
    if "}" in translation and len(translation.split("}")[1]) > 0:
        translation = translation.split("}")[0] + "}"
    try:
        json.loads(translation)
        return translation
    except Exception as e:
        raise e
    
def attempt_fix(example):
    try:
        return fix1(example)
    except Exception as e:
        try:
            return fix2(example)
        except Exception as e:
            try:
                return fix3(example)
            except Exception as e:
                raise e


def translate_example(example, random_seed=False, depth=0):
    ex = {
        "activity_label": example["activity_label"],
        "context": example["ctx"],
        "endings": example["endings"]
    }

    try:
        json_input = json.dumps(ex)
        if random_seed:
            out = structure_program(
                input=json_input,
                cache_seed=random.randint(0, 100000)
            )
        else:
            out = structure_program(
                input=json_input
            )
    except Exception as e:
        example["activity_label_de"] = ""
        example["ctx_de"] = ""
        example["endings_de"] = ["", "", "", ""]
        example["translation_de"] = ""
        return example
    try:
        translated = json.loads(out["output"])
        example["activity_label_de"] = translated["activity_label"]
        example["ctx_de"] = translated["context"]
        example["endings_de"] = translated["endings"]
        example["translation_de"] = out["output"]
    except Exception as e:
        try:
            translated = json.loads(attempt_fix(out))
            example["activity_label_de"] = translated["activity_label"]
            example["ctx_de"] = translated["context"]
            example["endings_de"] = translated["endings"]
            example["translation_de"] = out["output"]
        except Exception as e:
            if depth < 5:
                return translate_example(example, random_seed=True, depth=depth+1)
            example["activity_label_de"] = ""
            example["ctx_de"] = ""
            example["endings_de"] = ["", "", "", ""]
            example["translation_de"] = out["output"] if "output" in out else ""
    return example


dataset = load_dataset("hellaswag", split={"train": "train[:1000]", "validation": "validation"})


output_dir = Path("outputs_hellaswag_de")
output_dir.mkdir(exist_ok=True)
num_shards = 100
for split in ["train", "validation"]:
    ds = dataset[split]
    for i in trange(num_shards, desc=f"Translating {split} shards"):
        shard = ds.shard(num_shards=num_shards, index=i)
        shard = shard.map(translate_example, num_proc=32)
        shard.to_json(output_dir / f"{split}-{i:03d}.json")

# Combine shards

json_files = {
    "train": [str(x) for x in output_dir.glob(f"train-*.json")],
    "validation": [str(x) for x in output_dir.glob(f"validation-*.json")]
}
dataset = load_dataset("json", data_files=json_files)
# dataset.to_json(output_dir / "hellaswag_de.json")
dataset.push_to_hub("bjoernp/hellaswag_de")

for split in ["train", "validation"]:
    ds = dataset[split]
    # count examples with empty translation
    empty = ds.filter(lambda x: x["translation_de"] == "")
    print(f"Empty translations in {split}: {len(empty)}")
    # count examples with context translation
    empty = ds.filter(lambda x: x["ctx_de"] == "")
    print(f"Empty context translations in {split}: {len(empty)}")

