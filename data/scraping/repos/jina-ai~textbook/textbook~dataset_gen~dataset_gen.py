import threading
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import time

from typing import Callable, List, Protocol

import openai
from openai import OpenAIError

from pydantic import BaseModel
from textbook.dataset_gen.create_prompts import Topic
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TextColumn,
)
import hashlib

THREAD_LOCK = threading.Lock()
PROMPT_TOKENS_CNT = 0
COMPLETION_TOKENS_CNT = 0


class Exercise(BaseModel):
    problem: str
    solution: str


class Result(BaseModel):
    prompt: str
    output: str


def split_exercises(output: str) -> List[str]:
    """Split the result of the generation into separate functions"""
    return ["def" + i for i in output.split("def")[1:]]


def check_exercise(exercise: str) -> bool:
    try:
        if (
            "return" not in exercise.split('"""')[2]
            and "print" not in exercise.split('"""')[2]
        ):
            return False
        else:
            return True
    except IndexError:
        return False


def generator_to_exercises(output: str) -> List[Exercise]:
    exercises = split_exercises(output)
    exercises = [i for i in exercises if check_exercise(i)]
    results = []
    for j in exercises:
        try:
            splitted_exercise = j.split('"""')
            question = '"""'.join(splitted_exercise[:2]) + '"""'
            answer = splitted_exercise[2]
            results.append(Exercise(problem=question, solution=answer))
        except IndexError:
            splitted_exercise = j.split("'''")
            question = "'''".join(splitted_exercise[:2]) + "'''"
            answer = splitted_exercise[2]
            results.append(Exercise(problem=question, solution=answer))

    return results


class Generator(Protocol):
    def generate(self, prompt: str) -> Result:
        ...


class OpenAIGenerator:
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
    ):
        self.model = model

    def generate(self, prompt: str) -> Result:
        global PROMPT_TOKENS_CNT
        global COMPLETION_TOKENS_CNT
        chat_completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            timeout=60,
        )
        with THREAD_LOCK:
            PROMPT_TOKENS_CNT += chat_completion.usage.prompt_tokens
            COMPLETION_TOKENS_CNT += chat_completion.usage.completion_tokens
        result = Result(
            prompt=prompt, output=chat_completion.choices[0].message.content
        )

        return result


class GenerationError(OpenAIError):
    ...


class MonkeyGenerator:
    """
    A generator with a random response time and a random failure rate
    """

    def __init__(self, speed: int = 2, n_functions: int = 10):
        self.speed = speed
        self.n_functions = n_functions

    def generate(self, prompt: str) -> Result:
        seed = random.randint(0, 100)

        if self.speed > 0:
            time.sleep(seed / 100 * self.speed)
        # if not (seed % 50):
        #     raise GenerationError("Monkey failed")

        return Result(
            prompt=prompt,
            output='def gorilla(): """Empty function for a gorilla""" return 0'
            * self.n_functions,
        )


def generation(
    prompt: str,
    generator: Generator,
    update_progress: Callable,
    retries: int,
) -> List[Exercise]:
    success = False
    time.sleep(random.random())
    for i in range(retries):
        try:
            result = generator.generate(prompt)
            success = True
        except GenerationError:
            print(f"Generation failed for prompt {prompt}, retrying {i + 1}/{retries}")
            time.sleep(1)
        else:
            break

    if success:
        exercises = generator_to_exercises(result.output)
        update_progress()
        return exercises

    else:
        print(f"Generation failed for prompt {prompt}, skipping")
        return [Exercise(problem=prompt, solution="")]


def _generation_wrapper(
    prompt: str,
    get_generator: Callable[[], Generator],
    update_progress: Callable,
    save_dir: str,
    retries: int,
):
    file_path_sum = hashlib.md5(prompt.encode("utf-8")).hexdigest()

    dir_path, file_path = file_path_sum[:4], file_path_sum[4:]
    dir_path = os.path.join(save_dir, dir_path)
    file_path = os.path.join(dir_path, file_path + ".jsonl")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.exists(file_path):  # we don't regenerate each query
        print(f"skip {file_path} generation because it already exist ")
        return

    generator = get_generator()

    results = generation(prompt, generator, update_progress, retries)

    write_results_to_jsonl(file_path, results)


def mass_generation(
    prompts: List[str],
    get_generator: Callable[[], Generator],
    save_dir: str,
    pool_size: int = 10,
    retries: int = 10,
):
    """
    Generate from a list of prompts. Use a thread pool to parallelize the generation with catch and retry mechanism
    """
    with Progress(
        *Progress.get_default_columns(),
        "•",
        TimeElapsedColumn(),
        TextColumn("completion: [bold green]{task.fields[completion_tokens]}"),
        TextColumn("prompt: [bold green]{task.fields[prompt_tokens]}"),
    ) as progress:
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            progress_task = progress.add_task(
                "[red]Generating...",
                total=len(prompts),
                completion_tokens=0,
                prompt_tokens=0,
            )

            def update_progress():
                progress.update(
                    progress_task,
                    advance=1,
                    completion_tokens=COMPLETION_TOKENS_CNT,
                    prompt_tokens=PROMPT_TOKENS_CNT,
                )

            tasks = []

            for prompt in prompts:
                tasks.append(
                    executor.submit(
                        _generation_wrapper,
                        prompt,
                        get_generator,
                        update_progress,
                        save_dir,
                        retries,
                    )
                )

            for task in tasks:
                try:
                    task.result()
                except Exception as e:
                    print(e)


def load_prompts(file: str, key_prompt: str = "prompt") -> List[str]:
    with open(file, "r") as f:
        lines = f.readlines()

    prompts = [json.loads(line)[key_prompt] for line in lines]
    return prompts


def load_leaves(file: str) -> List[Topic]:
    with open(file, "r") as f:
        lines = json.load(f)
    topics = [Topic.parse_obj(line) for line in lines]
    return topics


def write_results_to_jsonl(file_path: str, results: List[Exercise]):
    with open(file_path, "w") as file:
        for item in results:
            json.dump(item.dict(), file)
            file.write("\n")
