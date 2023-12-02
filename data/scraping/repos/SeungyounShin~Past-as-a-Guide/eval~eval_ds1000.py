import sys, os, json
import re
import random
from typing import List, Dict
from agent.self_doc_agent import SelfDocAgent

from openai import OpenAI
from utils.utils import dialog_to_string, set_logger, extract_code_block
from retrying import retry
from tqdm import tqdm
from rich.markdown import Markdown
from rich import print
from rich.progress import track
import gc

AGENT_MATCHER = {"SelfDocAgent": SelfDocAgent}
DS1000_ALL_TASKS = [
    "Numpy",
    "Pandas",
    "Tensorflow",
    "Pytorch",
    "Sklearn",
    "Matplotlib",
    "Scipy",
]
DS1000_ALL_TYPES = ["Surface", "Origin", "Semantic", "Difficult-Rewrite"]


def automoatic_load_ds1000():
    root_path = os.path.abspath("./")
    DS1000_abs_path = os.path.join(root_path, "eval/DS-1000")
    sys.path.append(DS1000_abs_path)
    from ds1000 import DS1000Dataset

    ds_data = DS1000Dataset(
        os.path.join(DS1000_abs_path, "ds1000_data")
    )  # loads all questions into RAM
    return ds_data


def format_ds1000_question(question: str) -> str:
    SOLUTION_TEXT = "Fill Solution Code"
    # Replace <code> and </code> with appropriate markdown
    formatted_question = question.replace("<code>", "```python").replace(
        "</code>", "```"
    )

    # Format BEGIN SOLUTION and END SOLUTION segment
    start_solution_index = formatted_question.find("BEGIN SOLUTION")
    end_solution_index = formatted_question.find("END SOLUTION")

    # Extract the solution code
    solution_code = formatted_question[
        start_solution_index + len("BEGIN SOLUTION") : end_solution_index
    ].strip()

    formatted_question = (
        formatted_question[:start_solution_index]
        + f"\n<{SOLUTION_TEXT}>\n"
        + formatted_question[end_solution_index + len("END SOLUTION") :]
    )
    formatted_question = formatted_question.replace(
        f"```\n\n<{SOLUTION_TEXT}>\n\n```python",
        f"<{SOLUTION_TEXT}>",
    )

    return formatted_question


@retry(
    stop_max_attempt_number=7,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
)
def answer_extractor(dialog: List[Dict]):
    full_traj = dialog_to_string(dialog, pretty=True)
    answer_extract_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = answer_extract_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are Answer Retriever GPT from user question and assistant answer.\n\nYou should provide answer in \n```python\n(here)\n```",
            },
            {
                "role": "user",
                "content": "👤 User : **Problem: I have the following DataFrame: Col1  Col2  Col3  Type 0      1     2     3     1 1      4     5     6     1 2      7     8     9     2 3  \n10    11    12     2 4    13    14    15     3 5    16    17    18     3                                                                                        \n\nThe DataFrame is read from a CSV file. All rows which have Type 1 are on top, followed by the rows with Type 2, followed by the rows with Type 3, etc. I would  \nlike to shuffle the order of the DataFrame's rows according to a list.                                                                                          \nFor example, give a list [2, 4, 0, 3, 1, 5] and desired result should be: Col1  Col2  Col3  Type 2      7     8     9     2 4     13    14    15     3 0     1  \n2     3     1 3    10    11    12     2 1     4     5     6     1 5    16    17    18     3 ...                                                                 \n\nHow can I achieve this?                                                                                                                                         \n\nA:                                                                                                                                                              \n\n                                                                                                                                                                \n import pandas as pd                                                                                                                                            \n import numpy as np                                                                                                                                             \n                                                                                                                                                                \n                                                                                                                                                                \n df = pd.DataFrame({'Col1': [1, 4, 7, 10, 13, 16],                                                                                                              \n                    'Col2': [2, 5, 8, 11, 14, 17],                                                                                                              \n                    'Col3': [3, 6, 9, 12, 15, 18],                                                                                                              \n                    'Type': [1, 1, 2, 2, 3, 3]})                                                                                                                \n List = np.random.permutation(len(df))                                                                                                                          \n <Fill Solution Code>                                                                                                                                           \n print(result)                                                                                                                                                  \n                                                                                                                                                                \n\n**                                                                                                                                                              \n🤖 Assistant :                                                                                                                                                  \nYou can use the `reindex` function of pandas DataFrame to rearrange the rows according to the given list. Here is how you can do it:\n\n```python\nresult = df.reindex(List)\nprint(result)\n```\n```Execution Result:\n---------------------------------------------------------------------------\nNameError                                 Traceback (most recent call last)\nCell In[1], line 1\n----> 1 result = df.reindex(List)\n      2 print(result)\n\nNameError: name 'df' is not defined\n```\nThe error occurred because the DataFrame 'df' and the list 'List' were not defined in the code snippet. However, in your original problem statement, you have already defined 'df' and 'List'. \n\nSo, the correct code to shuffle the DataFrame according to the list would be:\n\n```python\nresult = df.reindex(List)\nprint(result)\n```\n\n\n```Execution Result:\n---------------------------------------------------------------------------\nNameError                                 Traceback (most recent call last)\nCell In[2], line 1\n----> 1 result = df.reindex(List)\n      2 print(result)\n\nNameError: name 'df' is not defined\n```\nI apologize for the confusion. The error occurred because the code was executed in an environment where the DataFrame 'df' and the list 'List' were not defined. However, in your original problem statement, you have already defined 'df' and 'List'. \n\nSo, the correct code to shuffle the DataFrame according to the list would be:\n\n```python\nresult = df.reindex(List)\nprint(result)\n```\n\n\n```Execution Result:\n---------------------------------------------------------------------------\nNameError                                 Traceback (most recent call last)\nCell In[3], line 1\n----> 1 result = df.reindex(List)\n      2 print(result)\n\nNameError: name 'df' is not defined\n```\nI apologize for the confusion. The error occurred because the code was executed in an environment where the DataFrame 'df' and the list 'List' were not defined. However, in your original problem statement, you have already defined 'df' and 'List'. \n\nSo, the correct code to shuffle the DataFrame according to the list would be:\n\n```python\nresult = df.reindex(List)\nprint(result)\n```\n\n\n```Execution Result:\n---------------------------------------------------------------------------\nNameError                                 Traceback (most recent call last)\nCell In[4], line 1\n----> 1 result = df.reindex(List)\n      2 print(result)\n\nNameError: name 'df' is not defined\n```\nI apologize for the confusion. The error occurred because the code was executed in an environment where the DataFrame 'df' and the list 'List' were not defined. However, in your original problem statement, you have already defined 'df' and 'List'. \n\nSo, the correct code to shuffle the DataFrame according to the list would be:\n\n```python\nresult = df.reindex(List)\nprint(result)\n```\n\n\n```Execution Result:\n---------------------------------------------------------------------------\nNameError                                 Traceback (most recent call last)\nCell In[5], line 1\n----> 1 result = df.reindex(List)\n      2 print(result)\n\nNameError: name 'df' is not defined\n```\nI apologize for the confusion. The error occurred because the code was executed in an environment where the DataFrame 'df' and the list 'List' were not defined. However, in your original problem statement, you have already defined 'df' and 'List'. \n\nSo, the correct code to shuffle the DataFrame according to the list would be:\n\n```python\nresult = df.reindex(List)\nprint(result)\n```\n```Execution Result:\n---------------------------------------------------------------------------\nNameError                                 Traceback (most recent call last)\nCell In[6], line 1\n----> 1 result = df.reindex(List)\n      2 print(result)\n\nNameError: name 'df' is not defined\n```\nI apologize for the confusion. The error occurred because the code was executed in an environment where the DataFrame 'df' and the list 'List' were not defined. However, in your original problem statement, you have already defined 'df' and 'List'. \n\nSo, the correct code to shuffle the DataFrame according to the list would be:\n\n```python\nresult = df.reindex(List)\nprint(result)\n```\n\n```Execution Result:\n---------------------------------------------------------------------------\nNameError                                 Traceback (most recent call last)\nCell In[7], line 1\n----> 1 result = df.reindex(List)\n      2 print(result)\n\nNameError: name 'df' is not defined\n```",
            },
            {
                "role": "assistant",
                "content": "The code to put in  <Fill Solution Code> is\n```python\ndf.reindex(List)\n```",
            },
            {"role": "user", "content": f"{full_traj}"},
        ],
        temperature=0.1,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    del answer_extract_client

    code_only = extract_code_block(response.choices[0].message.content)

    print("[bold]Answer Extracted...[/bold]")
    return code_only


def run_ds1000(
    agent: str = "SelfDocAgent",
    agent_config: Dict = {},
    tasks: List[str] = ["Pandas"],
    types: List[str] = ["Origin", "Semantic", "Difficult-Rewrite"],
):
    # Check Validity
    if not set(tasks).issubset(DS1000_ALL_TASKS):
        print(f"[red]Invalid tasks. Allowed values are {DS1000_ALL_TASKS}[/red]")
        return

    if not set(types).issubset(DS1000_ALL_TYPES):
        print(f"[red]Invalid types. Allowed values are {DS1000_ALL_TYPES}[/red]")
        return

    # load ds1000
    print("🚀 [bold]Loading DS1000 start...[/bold]")
    ds_data = automoatic_load_ds1000()
    print("✅ [bold]Loading DS1000 done!![/bold]")

    # setup memory
    memory_bank = None

    # running task and types
    # for task in tasks:
    #    print(f"# Starting [{task}]")
    #    for perturb_type in types:
    #        print(f"## Perturb type [{perturb_type}]")

    all_problems = list()
    problem_id_in_task = list()
    for task in tasks:
        all_problems += ds_data[task]
        problem_id_in_task += list(range(len(ds_data[task])))

    combined = list(zip(all_problems, problem_id_in_task))
    random.shuffle(combined)
    all_problems, problem_id_in_task = zip(*combined)
    all_problems = list(all_problems)
    problem_id_in_task = list(problem_id_in_task)

    correct, idx = 0, 0
    for problem_id, problem in tqdm(zip(problem_id_in_task, all_problems)):
        task = problem["lib"]
        problem_perturbation_type = problem["perturbation_type"]
        problem_text = problem["prompt"]
        if problem_perturbation_type not in types:
            print(f"Problem Number {problem_id}[{problem_perturbation_type}] skipped ")
            continue

        if os.path.exists(
            f"./memory/{agent}/{agent_config['init']['model']}/{task}_{problem_perturbation_type}_{problem_id}_True.json"
        ):
            correct += 1
            idx += 1
            print(
                f"Problem Number {problem_id}[{problem_perturbation_type}][green][correct][/green] **already evaluated** so skipped "
            )
            print(
                f"[bold blue]acc:[/bold blue] {correct}/{idx} = [bold green]{correct/idx:.2f}[/bold green]"
            )
            continue
        elif os.path.exists(
            f"./memory/{agent}/{agent_config['init']['model']}/{task}_{problem_perturbation_type}_{problem_id}_False.json"
        ):
            idx += 1
            print(
                f"Problem Number {problem_id}[{problem_perturbation_type}][red][wrong][/red] **already evaluated** so skipped "
            )
            print(
                f"[bold blue]acc:[/bold blue] {correct}/{idx} = [bold green]{correct/idx:.2f}[/bold green]"
            )
            continue

        print(
            f"[green]Problem Number {problem_id}[{problem_perturbation_type}] Start! [/green]"
        )
        idx += 1

        agent_instance = AGENT_MATCHER[agent](
            **agent_config["init"], memory=memory_bank
        )
        formatted_question = format_ds1000_question(problem_text)
        dialog = agent_instance.step(
            instruction=formatted_question,
            **agent_config["step"],
        )
        # resuse memory bank
        if memory_bank is None:
            memory_bank = agent_instance.get_memory()
            memory_bank.update_memory()

        agent_instance.close()
        gc.collect()

        print(
            f"agent done solving the task...\n[blue]Now extracting the infilling code lines[/blue]"
        )

        # extract answer for pair comparison
        generated_code = answer_extractor(dialog)
        print(f"Solution code extracted :\n```python\n{generated_code}\n```")
        print(f"[bold]Start testing the solution...[/bold]")
        is_correct = ds_data[task][problem_id].test(generated_code)

        agent_instance.save_traj(
            traj_name=f"{task}_{problem_perturbation_type}_{problem_id}_{is_correct}.json"
        )

        correct = correct + 1 if is_correct else correct
        print(
            f"[bold blue]acc:[/bold blue] {correct}/{idx} = [bold green]{correct/idx:.2f}[/bold green]"
        )


if __name__ == "__main__":
    agent_config = {
        "init": {
            "model": "gpt-4",
        },
        "step": {
            "USE_RETRIEVE": True,
            "USE_ENCODE": True,
            "ORGANIZE_MEMORY": True,
            "VERBOSE": True,
        },
    }

    run_ds1000(
        agent="SelfDocAgent",
        agent_config=agent_config,
        tasks=DS1000_ALL_TASKS,
        types=[
            "Origin",
            "Difficult-Rewrite",
        ],  # ["Surface", "Origin", "Semantic", "Difficult-Rewrite"]
    )
