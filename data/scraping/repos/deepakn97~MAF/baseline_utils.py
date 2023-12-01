import os
import json
import nltk
import numpy as np
import re
from langchain.prompts import (
    FewShotPromptTemplate,
    PromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
import itertools
from langchain.schema import HumanMessage
from io import StringIO
from contextlib import redirect_stdout
from src.utils import (
    Prompt,
    acall_gpt,
    call_gpt,
    VICUNA_MODEL_PATH,
    ALPACA_MODEL_PATH,
    OSModel,
    LLMModel,
    Logger,
)
from langchain.chat_models import ChatOpenAI
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.inference import load_model
import math
import torch
from src.entailment.utils import get_entailment_proof
from langchain.llms import OpenAI
import time
from typing import List
from tqdm import tqdm
from src.gsm_iter.gsm_selfref_eval import check_corr
import asyncio
from time import sleep
from typing import List, Dict, Union, Iterable, Tuple
import subprocess


OS_MODELS = ["vicuna", "alpaca"]
OPENAI_MODELS = ["gpt-3.5-turbo-0613", "gpt-3.5-turbo", "text-davinci-003", "gpt-4", 'gpt-4-0613']
TASKS = ["gsm_baseline", "entailment_baseline", "drop_baseline"]
PROMPTS = [
    "0cot_gsm",
    "1cot_gsm",
    "4cot_gsm",
    "pot_gsm",
    "ltm_gsm",
    "3shot_entailment",
    "4shot_entailment",
    "3shot_drop",
    "init_truncated",
]
QA_TEMPLATE = {
    "question_prefix": " # Q: ",
    "answer_prefix": "# A: ",
}
PYTHON_TEMPLATE = {
    "question_prefix": "# Q: ",
    "answer_prefix": "# solution using Python:\n",
}
ENTAILMENT_TEMPLATE = {
    "question_prefix": "",
    "answer_prefix": "Entailment Tree:\n",
    "intra_example_sep": "\n",
}
ZERO_TEMPLATE = {
    "instruction": 'Let\'s think step by step. End your answer with "final_answer: " and then the numeric answer to the question.',
    **QA_TEMPLATE,
}
TEMPLATES = {
    "qa": QA_TEMPLATE,
    "python": PYTHON_TEMPLATE,
    "entailment": ENTAILMENT_TEMPLATE,
    "zero": ZERO_TEMPLATE,
}


# I need to fix the memory allocation on OS Models
class BaselineWrapper:
    def __init__(
        self,
        engine: str = "text-davinci-003",
        task: str = "gsm_baseline",
        prompt: str = "pot_gsm",
        data_dir: str = None,
        save_dir: str = "src/baselines/models",
        llm=None,
        **kwargs,
    ):
        if task not in TASKS:
            raise ValueError(f"Invalid task {task}")
        self.task = task
        if prompt not in PROMPTS:
            raise ValueError(f"Invalid prompt {prompt}")
        self.prompt = prompt
        self.engine = engine
        if llm is not None:
            self.llm = llm
        else:
            if engine in OS_MODELS:
                self.llm = OSModel(engine=self.engine, **kwargs)
            elif engine in OPENAI_MODELS:
                self.llm = LLMModel(engine=self.engine, **kwargs)
            else:
                raise ValueError(f"Invalid engine {engine}")
        if task not in TASKS:
            raise ValueError(f"Invalid task {task}")
        if prompt not in PROMPTS:
            raise ValueError(f"Invalid prompt {prompt}")
        if data_dir is None:
            if task == "gsm_baseline":
                self.data_dir = "data/gsm_data"
            elif task == "entailment_baseline":
                self.data_dir = "data/entailment_data/baseline_data"
            elif task == "drop_baseline":
                self.data_dir = "data/drop_data/baseline_data"
            else:
                raise ValueError(f"Invalid task {task}")
        else:
            self.data_dir = data_dir
        self.save_dir = save_dir
        self.results_filepaths = []
        self.logger = Logger("src/baselines/baseline_log.txt")

    def run_batch(self, data, save_file, batch_size=None):
        if batch_size is None:
            outputs = self.llm([parse_problem(d, self.task) for d in data])
        else:
            outputs = []
            for i in tqdm(range(0, len(data), batch_size)):
                problems = [
                    parse_problem(d, self.task)
                    for d in data[i : min(i + batch_size, len(data))]
                ]
                outputs.extend(self.llm(problems))
                output_data = [{**d, "output": o} for d, o in zip(data, outputs)]
                with open(save_file, "w") as f:
                    f.write(json.dumps(output_data, indent=4) + "\n")
        output_data = [{**d, "output": o} for d, o in zip(data, outputs)]
        return output_data

    def run(
        self,
        parse=True,
        grade=True,
        batch_size=10,
        data=None,
        save_file=None,
        num_problems=None,
    ):
        prompt_file = f"prompt/{self.task}/{self.prompt}.json"
        if not os.path.exists(prompt_file):
            prompt_file = f"prompt/{self.task}/{self.prompt}.txt"
        self.llm.setup_prompt_from_examples_file(prompt_file)
        save_dir = os.path.join(
            self.save_dir, f"{self.engine}/{self.task}/{self.prompt}"
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if data is not None:
            self.logger.log(f"Running data (NOT RUNNING FROM DATA DIRECTORY)")
            if save_file is None:
                raise ValueError(
                    f"Invalid save file {save_file} (must pass explicitly if passing data explicitly)"
                )
            self.logger.log(f"Saving to {save_file}")
            if save_file not in self.results_filepaths:
                self.results_filepaths.append(save_file)
            if num_problems is not None:
                data = data[:num_problems]
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            data = self.run_batch(data, save_file, batch_size=batch_size)
            data = get_relevant_fields(data, self.task)
            with open(save_file, "w") as f:
                f.write(json.dumps(data, indent=4) + "\n")
        else:
            for data_file in os.listdir(self.data_dir):
                if not data_file.endswith(".jsonl"):
                    continue
                data = load_jsonl(os.path.join(self.data_dir, data_file))
                if num_problems is not None:
                    data = data[:num_problems]
                if self.task == "drop_baseline":
                    # for each passage, go through all the qa pairs and make them a separate question
                    data_expand = []
                    for d in data:
                        for qa in d["qa_pairs"]:
                            data_expand.append(
                                {
                                    "id": d["id"],
                                    "passage": d["passage"],
                                    "question": qa["question"],
                                    "answer": qa["answer"],
                                }
                            )
                    data = data_expand
                self.logger.log(f"Running {data_file}")
                self.logger.log(
                    "Example prompt: "
                    + self.llm.make_query(parse_problem(data[0], self.task))
                )
                save_file = os.path.join(
                    save_dir, data_file.replace(".jsonl", "_results.json")
                )
                self.logger.log(f"Saving to {save_file}")
                if save_file not in self.results_filepaths:
                    self.results_filepaths.append(save_file)
                data = self.run_batch(data, save_file, batch_size=batch_size)
                data = get_relevant_fields(data, self.task)
                with open(save_file, "w") as f:
                    f.write(json.dumps(data, indent=4) + "\n")
        if parse:
            self.parse_answers()
        if grade:
            self.grade_answers()

    def parse_answers(self):
        for filepath in self.results_filepaths:
            parse_answers(filepath, self.task, self.prompt)

    def grade_answers(self):
        for filepath in self.results_filepaths:
            grade_answers(filepath, self.task)


def eval_entailment(
    filepath,
    entailment_task,
    split="dev",
    bleurt_checkpoint="/data4/d_wang/nlp/models/bleurt-large-512",
    cleanup=True,
):
    # assert filepath.find(entailment_task) != -1
    filepath = os.path.abspath(filepath)
    # convert filepath to TSV
    with open(filepath, "r") as f:
        data = json.load(f)
    data = [d["answer"] for d in data]
    data = "\n".join(data)
    tsv_filepath = filepath.replace(".json", ".tsv")
    with open(tsv_filepath, "w") as f:
        f.write(data)
    # run eval script
    if entailment_task not in ["task_1", "task_2"]:
        raise ValueError(
            f"Invalid entailment task {entailment_task} (should be task_1 or task_2)"
        )
    eval_script = "eval/run_scorer.py"
    eval_cmd = f"conda run -n entbank python {eval_script}".split()
    eval_cmd.extend(
        [
            "--task",
            entailment_task,
            "--output_dir",
            os.path.dirname(filepath),
            "--split",
            split,
            "--prediction_file",
            tsv_filepath,
            "--bleurt_checkpoint",
            bleurt_checkpoint,
        ]
    )
    subprocess.run(eval_cmd, cwd="data/entailment_data")
    for postfix in [".json", ".metrics.json", ".diagnostics.tsv"]:
        os.rename(
            os.path.join(os.path.dirname(filepath), f"scores-{split}{postfix}"),
            os.path.join(
                os.path.dirname(filepath), f"scores-{entailment_task}-{split}{postfix}"
            ),
        )
    if cleanup:
        os.remove(tsv_filepath)


def nested_get_pair(d, keys, suppress_error=False):
    try:
        if not isinstance(keys, Iterable) or type(keys) == str:
            return d[keys]
        else:
            if len(keys) == 1:
                return d[keys[0]]
            else:
                res = nested_get_pair(d[keys[0]], keys[1:])
                return res
    except KeyError:
        if suppress_error:
            return None
        else:
            raise KeyError(f"Key {keys} not found in {d}")


def resolve_tuple_keys(d):
    if isinstance(d, dict):
        items = list(d.items())
        for k, v in items:
            if isinstance(k, tuple):
                if len(k) == 1:
                    d[k[0]] = v
                else:
                    d[k[0]] = resolve_tuple_keys({k[1:]: v})
                del d[k]
            else:
                d[k] = resolve_tuple_keys(v)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            d[i] = resolve_tuple_keys(v)
    return d


def get_relevant_fields(data, task):
    FIELDS = {
        "common": ["output", "answer", "correct"],
        "gsm_baseline": ["input", "target", "n_steps"],
        "entailment_baseline": [
            "id",
            "hypothesis",
            "solution",
            "proof",
            ("meta", "triples"),
        ],
        "drop_baseline": [
            "id",
            "passage",
            "question",
            "final_answer",
        ],
    }
    if task not in TASKS:
        raise ValueError(f"Invalid task {task}")
    fields = FIELDS["common"] + FIELDS[task]
    try:
        res = [{k: nested_get_pair(d, k, True) for k in fields} for d in data]
        res = resolve_tuple_keys(res)
        return res
    except KeyError as e:
        print(f"Key error when trying to parse {task} data. Missing key {e}")


def parse_problem(problem_dict, task):
    if task not in TASKS:
        raise ValueError(f"Invalid task {task}")
    if task == "gsm_baseline":
        return problem_dict["input"]
    elif task == "entailment_baseline":
        p = "Hypothesis: " + problem_dict["hypothesis"] + "\n\n" + "Text:"
        for sent, text in problem_dict["meta"]["triples"].items():
            p += "\n" + sent + ": " + text
        return p
    elif task == "drop_baseline":
        p = "Passage: " + problem_dict["passage"] + "\n\n"
        # TODO: don't assume we are only using first question later on
        p += "Q: " + problem_dict["question"] + "\n\n"
        return p


def grade_answers(filepath, task, split=None, overwrite=True):
    with open(filepath, "r") as f:
        data = json.loads(f.read())
    if task == "gsm_baseline":
        for d in data:
            d["correct"] = check_corr(d["answer"], d["target"], task)
        # return sum([d["correct"] for d in data]) / len(data)
    elif task == "entailment_baseline":
        # assert that filepath is in the format task_{number}_{split}_results.json
        assert re.match(
            r"task_(\d)_(dev|train|test)_results.json", os.path.basename(filepath)
        )
        filename = os.path.basename(filepath)
        if split is None:
            split = filename.split("_")[-2]
        entailment_task = f"task_{filename.split('_')[1]}"
        eval_entailment(filepath, entailment_task, split)
    elif task == "drop_baseline":
        for d in data:
            d["correct"] = check_corr_drop(d["final_answer"], d["answer"])
    else:
        raise ValueError(f"Invalid task {task}")
    if not overwrite:
        filepath = os.path.join(
            os.path.dirname(filepath), "graded_" + os.path.basename(filepath)
        )
    with open(filepath, "w") as f:
        f.write(json.dumps(data, indent=4) + "\n")
    # return accuracy


def parse_answers(filepath, task, prompt_technique, overwrite=True):
    with open(filepath, "r") as f:
        problems = json.load(f)
    final_answer_key = ""
    if task in ["gsm_baseline", "entailment_baseline"]:
        final_answer_key = "answer"
    elif task == "drop_baseline":
        final_answer_key = "final_answer"
    for p in problems:
        p[final_answer_key] = parse_answer(p["output"], task, prompt_technique)
    with open(filepath, "w") as f:
        f.write(json.dumps(problems, indent=4) + "\n")


def load_jsonl(filepath):
    """Loads jsonl file into list of dict objects"""
    if (
        not os.path.exists(filepath)
        or not os.path.isfile(filepath)
        or not filepath.endswith(".jsonl")
    ):
        raise ValueError(f"Invalid jsonl filepath {filepath}")
    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def create_prompt_template(task, model_name, prompt_technique):
    with open(
        os.path.join(f"prompt/{task}/{model_name}", f"{prompt_technique}.txt"), "r"
    ) as f:
        template = f.read()

    prompt = PromptTemplate(input_variables=["context"], template=template)
    return prompt


def calc_accuracy(problems):
    return sum([p["correct"] for p in problems]) / len(problems)


def check_corr(input: str, target: str, task: str, tol: float = 0.001):
    if task not in TASKS:
        raise ValueError(f"Invalid task {task}")
    if task == "gsm_baseline":
        try:
            return abs(float(input) - float(target)) < tol
        except:
            return False
    elif task == "entailment_baseline":
        raise NotImplementedError()


def check_corr_drop(input: str, target: dict[str, str | dict[str, str]]):
    # only one of number, date, and spans will be nonempty.
    if target["number"] != "":
        # remove units from input
        input = re.sub(r"[a-zA-Z]", "", input)
        return input == target["number"].lower()
    elif (
        target["date"]["day"] != ""
        or target["date"]["month"] != ""
        or target["date"]["year"] != ""
    ):
        items = []
        for key in target["date"]:
            if target["date"][key] != "":
                items.append(target["date"][key])
        # get all permutations of items
        perms = list(itertools.permutations(items))
        # check if any of the permutations are equal to input
        return any(["-".join(p) == input for p in perms])
    elif target["spans"] != "":
        for span in target["spans"]:
            if span.lower() not in input.lower():
                return False
        return True
    else:
        raise ValueError("No target answer found in 'number', 'date', or 'spans'")


def manual_parse(filename):
    with open(filename, "r") as f:
        problems = json.load(f)
    for problem in problems:
        # check if target is anywhere in output
        if problem["target"] in problem["output"]:
            problem["final_answer"] = "undefined"
        else:
            problem["final_answer"] = "wrong"
    with open(filename, "w") as f:
        f.write(json.dumps(problems, indent=4))


def parse_program(answer):
    # Takes code that ends in print statement. Returns output of print statement
    answer = answer.split("\n")
    # if there is a line with def solution(), only take the lines with and after the function
    for i, line in enumerate(answer):
        if line.lower().find("def solution") != -1:
            answer = answer[i:]
            break
    # if there is no line with def solution(), add one to beginning
    if answer[0].lower().find("def solution") == -1:
        answer = ["def solution():"] + answer
    # if there is a return statement, replace with print statement
    for i, line in enumerate(answer):
        if line.lower().find("return") != -1:
            result = line.split("return")[1]
            answer[i] = line.split("return")[0] + f"print({result})"
            break
    # Remove lines after print statement
    for i, line in enumerate(answer):
        if line.lower().find("print") != -1:
            answer = answer[: i + 1]
            break
    # add a line to run solution()
    answer.append("solution()")
    return "\n".join(answer)


def parse_program_answer(answer):
    # Execute code and capture print output
    answer = parse_program(answer)
    f = StringIO()
    try:
        with redirect_stdout(f):
            exec(answer)
        answer = f.getvalue()
        answer = answer.split("\n")[0]
        answer = "".join(c for c in answer if c.isdigit() or c == ".")
        return answer
    except:
        return "undefined"


def parse_answer(answer: str, task: str, prompt_technique: str):
    if task == "gsm_baseline":
        if prompt_technique == "pot_gsm":
            return parse_program_answer(answer)
        else:
            answer_original = answer.lower()
            for answer_key in ["final_answer: ", "final answer: ", "final answer is: "]:
                answer = answer_original
                answer = answer[answer.find(answer_key) + len(answer_key) :]
                answer = answer.split("\n")[0]
                answer = "".join(
                    c for c in answer if c.isdigit() or c == "."
                )  # note that commas are removed
                try:
                    fl_answer = float(answer)
                    int_answer = int(fl_answer)
                    if fl_answer == int_answer:
                        return str(int_answer)
                    else:
                        return str(fl_answer)
                except:
                    return "undefined"
            return "undefined"
    elif task == "entailment_baseline":
        answer = get_entailment_proof([answer])[0]
        # Split by lines, only take lines including and after 'Entailment Tree:'
        return "$proof$ = " + answer
    elif task == "drop_baseline":
        answer_original = answer.lower()
        for answer_key in ["final_answer: ", "final answer: ", "final answer is: "]:
            if answer_key not in answer_original:
                continue
            answer = answer_original
            answer = answer[answer.find(answer_key) + len(answer_key) :]
            answer = answer.split("\n")[0]
            answer = answer.strip()
            return answer
        return "undefined"


def to_tsv(filepath, lines):
    with open(filepath, "w") as f:
        for line in lines:
            f.write(line + "\n")
