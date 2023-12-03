import json
import uuid
import os
from itertools import islice
from glob import glob
from multiprocessing import Pool
from gen_prompts.parsers import (
    PromptDetector,
    used_langchain_tool,
    used_chat_function,
    used_in_langchain_llm_call,
    used_in_openai_call,
    prompt_or_template_in_name,
    new_line_in_string,
    all_strings,
)

RUN_ID = f"{5:03d}"
os.makedirs(RUN_ID, exist_ok=True)


def process_chunk(filenames):
    detector = PromptDetector()
    detector.add_heuristic(used_langchain_tool)
    detector.add_heuristic(used_in_langchain_llm_call)
    detector.add_heuristic(used_in_openai_call)
    detector.add_heuristic(used_chat_function)
    detector.add_heuristic(prompt_or_template_in_name)
    # detector.add_heuristic(new_line_in_string)
    # detector.add_heuristic(all_strings)

    detector.detect_prompts(filenames)
    # _uuid = str(uuid.uuid4())
    # with open(f"{RUN_ID}/prompts-{_uuid}.json", "w") as w:
    #     json.dump(prompts, w)


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def run_all(filenames: list[str], n=32):
    filenames_batched = batched(filenames, len(filenames) // n)
    with Pool(n) as p:
        p.map(process_chunk, filenames_batched)


def single_run(filenames: list[str], n=32):
    process_chunk(filenames)


if __name__ == "__main__":
    root_dir = "data/scraping/repos"

    paths = []
    for repo in os.listdir(root_dir):
        repo_path = os.path.join(root_dir, repo)
        for file in os.listdir(repo_path):
            file_path = os.path.join(repo_path, file)
            paths.append(file_path)

    single_run(paths)

    data = []
    for filename in glob(f"{RUN_ID}/prompts-*.json"):
        with open(filename) as f:
            data.extend(json.load(f))

    print(f"Found {len(data)} prompts in {len(paths)} files.")
    with open(f"final_prompts_{RUN_ID}.json", "w") as file:
        json.dump(data, file)
