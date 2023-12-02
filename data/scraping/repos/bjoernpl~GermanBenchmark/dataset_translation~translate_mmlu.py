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
from tqdm import tqdm
import guidance
import json
from pathlib import Path

_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]
                                                      
# set the default language model used to execute guidance programs
guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo", max_calls_per_min=1000, api_key=input("API Key: ")) 

structure_program = guidance(
'''
{{#system~}}
You are a helpful assistant that translates questions and answers from English to German.
{{~/system}}

{{#user~}}
Translate the following question and each of the multiple choice answers into German. Be as precise as possible. Keep the exact json format. 
Translate only the values and not the keys. "_________" indicate blanks that should be kept in the translation. Do not answer anything else than the json.

{
"question": "How many planets does our solar system have?",
"A": "8",
"B": "9",
"C": "10",
"D": "All of the above"
}
{{~/user}}

{{#assistant~}}
{
"question": "Wie viele Planeten hat unser Sonnensystem?",
"A": "8",
"B": "9",
"C": "10",
"D": "Alle der oben genannten"
}
{{~/assistant}}

{{#user~}}
{
"question": {{input}},
"A": "{{a}}",
"B": "{{b}}",
"C": "{{c}}",
"D": "{{d}}"
}
{{~/user}}

{{#assistant~}}
{{gen 'output' temperature=0 top_p=1 stop="\\n}" max_tokens=1500}}
{{~/assistant}}
''', stream=False)

json_format = """
"question": "{question}",
"A": "{A}",
"B": "{B}",
"C": "{C}",
"D": "{D}"
"""

def contains_json(string):
    return "{" in string and "}" in string and "\"question\"" in string and "\"A\"" in string and "\"B\"" in string and "\"C\"" in string and "\"D\"" in string

def fix_quotes(string):
    if string[0] == "\"":
        string = string[1:]
    if string[-1] == "\"":
        string = string[:-1]
    string = string.replace("\"", "\\\"")
    string = string.replace("\n", "\\n")
    return string

def fix_parentheses(string):
    string = string.replace("{", "\\{")
    string = string.replace("}", "\\}")
    return string

def get_question(string):
    post_question = string.split("\"question\":")[1]
    question = post_question.split("\"A\"")[0].strip()
    if question[0] == "\"":
        question = question[1:]
    if question[-2:] == "\",":
        question = question[:-2]
    question = question.replace("\"", "\\\"")
    question = question.replace("\n", "\\n")
    question = question.replace("\\\",\\n\\\"", "\\n")
    question = fix_parentheses(question)
    return question

def get_choices(string):
    choice_A = string.split("\"A\":")[1].split("\"B\"")[0].strip()[:-1]
    choice_B = string.split("\"B\":")[1].split("\"C\"")[0].strip()[:-1]
    choice_C = string.split("\"C\":")[1].split("\"D\"")[0].strip()[:-1]
    choice_D = string.split("\"D\":")[1].split("}")[0].strip()
    if choice_D.endswith(","):
        choice_D = choice_D[:-1]
    fix = lambda x: fix_quotes(fix_parentheses(x))
    return [fix(choice_A), fix(choice_B), fix(choice_C), fix(choice_D)]


def is_valid_json(string):
    try:
        json.loads(string)
        return True
    except:
        return False
    
def get_json(string):
    if contains_json(string):
        question = get_question(string)
        choices = get_choices(string)
        json_string = json_format.format(question=question, A=choices[0], B=choices[1], C=choices[2], D=choices[3])
        json_string = "{" + json_string + "}"
        if is_valid_json(json_string):
            return json_string
        else:
            return None
    else:
        return None

mmlu = {name: load_dataset("tasksource/mmlu", name, split="validation") for name in _SUBJECTS}
total_len = sum(len(mmlu[name]) for name in _SUBJECTS)
print(f"Total length: {total_len} examples")

def translate_example(example):
    question = example["question"]
    try:
        out = structure_program(
            input=question,
            a=example["choices"][0],
            b=example["choices"][1],
            c=example["choices"][2],
            d=example["choices"][3]
        )
    except:
        example["answer_de"] = out["output"]
        example["question_de"] = ""
        example["choices_de"] = ["", "", "", ""]
    try:
        translated = json.loads(get_json(out["output"]+"\n}"))
        example["question_de"] = translated["question"]
        example["choices_de"] = [translated["A"], translated["B"], translated["C"], translated["D"]]
        example["answer_de"] = out["output"]+"\n}"
    except:
        if "{" in out["output"]:
            output = "{" + out["output"].split("{")[1]
            try:
                translated = json.loads(output)
                example["question_de"] = translated["question"]
                example["choices_de"] = [translated["A"], translated["B"], translated["C"], translated["D"]]
                example["answer_de"] = out["output"]+"\n}"
            except:
                example["answer_de"] = out["output"]
                example["question_de"] = ""
                example["choices_de"] = ["", "", "", ""]
        else:
            example["answer_de"] = out["output"]
            example["question_de"] = ""
            example["choices_de"] = ["", "", "", ""]
    return example

Path("outputs_val_mmlu").mkdir(exist_ok=True)
# Translate the parts
for i, name in tqdm(enumerate(_SUBJECTS), total=len(_SUBJECTS)):
    print(f"Translating {name} ({i+1}/{len(_SUBJECTS)})")
    part = mmlu[name]
    part = part.select(range(15)) if len(part) > 15 else part
    p = part.map(translate_example, num_proc=8)
    p = p.filter(lambda x: x["question_de"] != "")
    print(len(p) > 6)
    p.to_parquet(f"outputs_val_mmlu/{name}.parquet")

