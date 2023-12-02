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
It consists of a question and multiple possible answers.
Be as precise as possible. Keep the exact json format and do not translate the keys.

{{input}}
{{~/user}}

{{#assistant~}}
{{gen 'output' temperature=1 top_p=1}}
{{~/assistant}}
''', stream=False)

question_options = ["question", "Frage", "frage"]
choices_options = ["choices", "Antworten", "Antwortmöglichkeiten", "Auswahlmöglichkeiten", "Möglichkeiten", "Optionen", "Aussagen", "Auswahlen", "möglichkeiten", "optionen", "aussagen", "auswahlen", "antworten", "antwortmöglichkeiten", "auswahlmöglichkeiten", "Auswahl", "auswahl"]

def get_question_and_choices(example):
    question = None
    choices = None
    for q in question_options:
        if q in example:
            question = example[q]
            break
    for c in choices_options:
        if c in example:
            choices = example[c]
            break
    return question, choices

def manual_fix(translation):
    print(translation)
    print("Please enter the correct translation:")
    new = input()
    try:
        json.loads(new)
        return new
    except Exception as e:
        print("Invalid json, please try again")
        return manual_fix(translation)


def translate_example(example, mc1=True):
    targets = "mc1_targets" if mc1 else "mc2_targets"
    other = "mc2_targets" if mc1 else "mc1_targets"
    ex = {
        "question": example["question"],
        "choices": example[targets]["choices"]
    }

    try:
        json_input = json.dumps(ex)
        out = structure_program(
            input=json_input
        )
    except Exception as e:
        example["question_de"] = example.get("question_de", "")
        example[targets+"_de"] = {"choices": [""]*len(example[targets]["choices"]), "labels": example[targets]["labels"]}
        example[other+"_de"] = example.get(other+"_de", {"choices": [""]*len(example[other]["choices"]), "labels": example[other]["labels"]})
        example["translation_de"+ ("1" if mc1 else "2")] = ""
        example["translation_de"+ ("2" if mc1 else "1")] = example.get("translation_de"+ ("2" if mc1 else "1"), "")
        print("first exception")
        return example
    try:
        try:
            translated = json.loads(out["output"])
        except Exception as e:
            translated = json.loads(manual_fix(out["output"]))
        question, choices = get_question_and_choices(translated)
        if question is None or choices is None:
            print(translated.keys())
        if question is None:
            question = ""
        if choices is None:
            choices = [""]*len(example[targets]["choices"])
        example["question_de"] = question
        example[targets+"_de"] = {"choices": choices, "labels": example[targets]["labels"]}
        example[other+"_de"] = example.get(other+"_de", {"choices": [""]*len(example[other]["choices"]), "labels": example[other]["labels"]})
        example["translation_de"+ ("1" if mc1 else "2")] = out["output"]
        example["translation_de"+ ("2" if mc1 else "1")] = example.get("translation_de"+ ("2" if mc1 else "1"), "")
    except Exception as e:
        example["question_de"] = example.get("question_de", "")
        example[targets+"_de"] = {"choices": [""]*len(example[targets]["choices"]), "labels": example[targets]["labels"]}
        example[other+"_de"] = example.get(other+"_de", {"choices": [""]*len(example[other]["choices"]), "labels": example[other]["labels"]})
        example["translation_de"+ ("1" if mc1 else "2")] = out["output"] if "output" in out else ""
        example["translation_de"+ ("2" if mc1 else "1")] = example.get("translation_de"+ ("2" if mc1 else "1"), "")
    return example


dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")


output_dir = Path("outputs_truthfulqa_de")
output_dir.mkdir(exist_ok=True)
num_shards = 10
for i in trange(num_shards, desc=f"Translating shards"):
    shard = dataset.shard(num_shards=num_shards, index=i)
    shard = shard.map(translate_example, num_proc=1)
    shard = shard.map(translate_example, num_proc=1, fn_kwargs={"mc1": False})
    shard.to_json(output_dir / f"{i:03d}.json")

# Combine shards

json_files = {
    "validation": [str(x) for x in output_dir.glob(f"*.json")]
}
dataset = load_dataset("json", data_files=json_files)["validation"]
dataset.push_to_hub("bjoernp/truthful_qa_de")

# count examples with empty translation
empty = dataset.filter(lambda x: x["translation_de1"] == "")
print(f"Empty translations1 in dataset: {len(empty)}")
empty = dataset.filter(lambda x: x["translation_de2"] == "")
print(f"Empty translations2 in dataset: {len(empty)}")
# count examples with question translation
empty = dataset.filter(lambda x: x["question_de"] == "")
print(f"Empty question translations in dataset: {len(empty)}")

empty = dataset.filter(lambda x: x["mc1_targets_de"]["choices"]==None)#["choices"][0] == "")
print(f"Empty mc1 translations in dataset: {len(empty)}")

empty = dataset.filter(lambda x: x["mc1_targets_de"]["choices"]!=None and x["mc1_targets_de"]["choices"][0] == "")
print(f"Empty mc1 translations in dataset: {len(empty)}")

empty = dataset.filter(lambda x: x["mc2_targets_de"]["choices"]==None)#["choices"][0] == "")
print(f"Empty mc2 translations in dataset: {len(empty)}")

empty = dataset.filter(lambda x: x["mc2_targets_de"]["choices"]!=None and x["mc2_targets_de"]["choices"][0] == "")
print(f"Empty mc2 translations in dataset: {len(empty)}")

