import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from human_eval.data import read_problems
from langchain.callbacks import get_openai_callback
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# CONFIG
DEBUG = True
PRINT_TOKEN_CONSUMPTION = False

TEMP_MIN = 0.0
TEMP_MAX = 1.0
TOP_P_MIN = 0.0
TOP_P_MAX = 1.0
STEP = 0.2

if DEBUG:
    STEP = 1


def get_result(samples_path):
    result = subprocess.run(
        [
            "evaluate_functional_correctness",
            samples_path,
            "1,10,100",
            "4",
            "3",
            "data/problems.jsonl",
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr


def eval_sample_problem_pair(samples_path):
    # Get result of running ".venv/bin/evaluate_functional_correctness"
    result, _ = get_result(samples_path)
    with open(f"{samples_path}_results.txt", "w") as f:
        f.write(result.splitlines()[-1])


def write_modified_problems(problems):
    """
    Write modified problems to data/problems.jsonl.
    The modification is because the GPT-4 model
    is a chat model that doesn't do well on strict sentence completion.
    So instead, we ask it to define the complete function.
    This requires slightly updated HumanEval problem jsonl lines.
    """
    # Print first problem, it is a dict like this:
    # Iterate over dict
    collect_problems = []

    for key, value in problems.items():
        # Create empty problem
        collect_problems.append(
            {
                "task_id": key,
                "prompt": "",
                "entry_point": value["entry_point"],
                "test": value["test"],
            }
        )

        # This is just validation, our logic doesn't work if no `def ` is in the prompt
        def_split = value["prompt"].split("def ")
        assert len(def_split) != 1, "Prompt must always contain a 'def '"

    # Write problems to JSONL with a single line for each entry in collect_problems
    # be sure to convert the dict to a single line string with json.dumps()
    with open("data/problems.jsonl", "w") as f:
        for problem in collect_problems:
            f.write(json.dumps(problem) + "\n")


# Modified to return the collect_samples dictionary
def generate_solution_for_problem(task_id, problem, chat):
    with get_openai_callback() as cb:
        message = chat(
            [
                SystemMessage(
                    content="You are an expert Python programmer. Implement the function provided by the user. Make sure your implementation is correct. Only output code since your output will directly be executed."
                ),
                HumanMessage(content=problem["prompt"]),
            ]
        )
        if PRINT_TOKEN_CONSUMPTION:
            print(cb)

    # Assuming no import code exists after first function definition
    # manual inspection of the HumanEval data showed this was the case.
    def_split = problem["prompt"].split("def ")
    imports_code = def_split[0]

    solution = imports_code + message.content

    return {
        "task_id": task_id,
        "completion": solution,
    }


def generate_solutions_for_params(temperature, top_p, problems):
    # Set max_retries really high, the concurrency will
    # cause us to hit the rate limit often.
    chat = ChatOpenAI(
        max_retries=1000,
        temperature=temperature,
        model_kwargs={
            "top_p": top_p,
        },
    )

    collect_samples = []

    # Generate solutions in parallel
    with ThreadPoolExecutor() as executor:
        future_to_task = {
            executor.submit(generate_solution_for_problem, key, value, chat): (
                key,
                value,
            )
            for key, value in problems.items()
        }

        for future in as_completed(future_to_task):
            collect_samples.append(future.result())

    with open(f"data/samples_{temperature:.2f}_{top_p:.2f}_.jsonl", "w") as f:
        for sample in collect_samples:
            f.write(json.dumps(sample) + "\n")


def generate_solutions():
    problems = read_problems()

    if DEBUG:
        # Only use the first two problems
        problems = {k: v for k, v in list(problems.items())[:2]}

    write_modified_problems(problems)

    combinations = [
        (temperature, top_p)
        for temperature in np.arange(TEMP_MIN, TEMP_MAX + STEP, STEP)
        for top_p in np.arange(TOP_P_MIN, TOP_P_MAX + STEP, STEP)
    ]

    # Generate solutions for all combinations in parallel
    with ThreadPoolExecutor() as executor:
        future_to_comb = {
            executor.submit(
                generate_solutions_for_params, temperature, top_p, problems
            ): (temperature, top_p)
            for temperature, top_p in combinations
        }

        for future in as_completed(future_to_comb):
            temperature, top_p = future_to_comb[future]
            print(f"Completed temperature: {temperature}, top_p: {top_p}")


def verify_generated_solutions():
    # List all JSONL files in the data directory
    jsonl_files = [f for f in os.listdir("data/") if f.endswith(".jsonl")]

    # Separate samples and problems files
    samples_files = [f for f in jsonl_files if "samples" in f and "results" not in f]

    for samples_file in samples_files:
        samples_path = os.path.join("data", samples_file)
        print(f"Running eval() for {samples_file}.")
        eval_sample_problem_pair(samples_path)


if __name__ == "__main__":
    generate_solutions()
    verify_generated_solutions()
