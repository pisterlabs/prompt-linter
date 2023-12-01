#!/usr/bin/env python
# coding: utf-8
import enum
import itertools as it
import logging
import pathlib
import time
import threading
import sys


import fire
import jsonlines as jsonl
import more_itertools as mit
import pathvalidate
import openai
import rich
import rich.table
import rich.markup
import rich.rule
import rich.status
from tqdm import tqdm


SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR / "Marg-Li-CoT/with_trl/approach_answer/"))
sys.path.append(str(SCRIPT_DIR / "Marg-Li-CoT/with_trl/"))
print(" - " + "\n - ".join(sorted(sys.path)))
import lib_data_commonsense_qa


DEFAULT_RETRY_UNTIL_GOOD = False
DEFAULT_NUM_RETRY_UNTIL_GOOD = 5
DEFAULT_TEMPERATURE_INCREASE = 0.05

DEFAULT_MODEL_NAME = "gpt-4" # "gpt-4-0613" # 
DEFAULT_PATH_SECRET = "/home/mila/g/gagnonju/openai/openai.txt"
DEFAULT_TIMEOUT_DURATION = 1
DEFAULT_RETRY_WAIT_TIME = 10


class CVSets(str, enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"


def _query_one_error_handler(error, thread_name, retry_wait_time):
    error_name = type(error).__name__
    header = f"{error_name} - {thread_name}"
    rich.print(
        f"\n[red on white]{header}:[/] {error}\n"
        f"[red on white]Waiting {retry_wait_time} seconds")
    time.sleep(retry_wait_time)
    rich.print(f"[green on white]Retrying.\n")


def _query_one(
    *, 
    model: str,
    num_retry_until_good: int,
    retry_wait_time: int,
    retry_until_good: bool,
    sample: dict[str, str], 
    temperature_increase: float,
    timeout_duration: int, 
):
  
    thread_name = threading.current_thread().getName()
    prompt = sample["ref_fs_scratchpad_gen_query"]
    num_retries_until_good_so_far = 0
    temperature = 0

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=100,
                timeout=timeout_duration,
                stop=["\n"],
            )
        except (
            openai.error.OpenAIError, 
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            TimeoutError,
        ) as e:
            _query_one_error_handler(e, thread_name, retry_wait_time)
            continue
    
        output = mit.one(response.choices)["message"]["content"]
        is_good = sample["ref_qa_answer"] in output
        
        if not is_good:
            if retry_until_good:
                table = rich.table.Table("Key", "Value", title="Incorrect answer", show_lines=True)
                table.add_row(f"[bold]Question",       sample['ref_qa_question'])
                table.add_row(f"[bold]Choices",        sample['ref_qa_choices'])
                table.add_row(f"[bold]ref_answer",     sample['ref_qa_answer'])
                table.add_row(f"[bold]model_answer",   output)
                table.add_row(f"[bold]Temperature",    f"{temperature}")                
                rich.print(table)

                rich.print(f"Retrying until good, {num_retries_until_good_so_far}/{num_retry_until_good} retries.")

                num_retries_until_good_so_far += 1
                temperature += temperature_increase

                if num_retries_until_good_so_far < num_retry_until_good:
                    continue

                rich.print(rich.rule.Rule())
        
        # If we get here, we're good.
        temperature = 0
        num_retries_until_good_so_far = 0
        break

    return dict(**sample, output=output), is_good


def _check_matches_setup_resuming(dataset_sample, saved_sample):
    for k in dataset_sample:
        assert dataset_sample[k] == saved_sample[k], (
            k, dataset_sample[k], saved_sample[k])


def _setup_resuming(*, data, output_path, split):

    with jsonl.open(output_path, "r") as f:
        existing = list(f)
    existing_dict = {x["ref_qa_id"]: x for x in existing}

    rich.print(f"[bold blue on white]({split}): Checking existing data.")
    for dataset_sample in it.islice(data, len(existing_dict)):
        saved_sample = existing_dict[dataset_sample["ref_qa_id"]]
        _check_matches_setup_resuming(
            dataset_sample=dataset_sample, 
            saved_sample=saved_sample,
        )
        
    rich.print(f"[bold green on white]({split}): Did {len(existing)} of {len(data)} already.")

    for dataset_sample, saved_sample in mit.zip_equal(
        it.islice(data, len(existing)), existing):
        
        _check_matches_setup_resuming(
            dataset_sample=dataset_sample, 
            saved_sample=saved_sample,
        )

    rich.print(f"[bold green on white]({split}): Did {len(existing)} of {len(data)} already.")

    return len(existing)


def inference_on_split(
        *, 
        give_model_answers,
        model: str,
        num_retry_until_good: int,
        path: pathlib.Path,
        retry_wait_time: int,
        retry_until_good: bool,
        split,
        test_run,
        temperature_increase,
        timeout_duration: int,
    ):
    
    print(f"{split}: {path}")
    data = lib_data_commonsense_qa.CommonSenseScratchpadGenMC(
        any_tokenizer=None, 
        give_model_answers=give_model_answers,
        split=split.value,
        text_only=True,
    )

    if path.exists():
        rich.print(f"[bold green on white]({split}): Resuming inference.")
        initial_count = _setup_resuming(
            output_path=path, 
            data=data, 
            split=split,
        )
    else:
        rich.print(f"[bold blue on white]({split}): Not resuming inference.")
        initial_count = 0

    def _description_template(goods, total): 
        if total: 
            fraction_str = f"{goods/total:0.2%}"
        else:
            fraction_str = "N/A"
        
        return f"({split}): Good: {goods}/{total} ({fraction_str})"
            
    
    rich.print(f"[bold blue on white]({split}): Starting inference.")
    if not test_run:
        with jsonl.open(path, "a", flush=True) as f:
            total = 0
            goods = 0
            
            progress = tqdm(
                it.islice(data, initial_count, None, 1),
                total=len(data),    
                initial=initial_count,
                desc=_description_template(goods, total),
            )
            for sample in progress:
                output, is_good = _query_one(
                    model=model,
                    retry_wait_time=retry_wait_time,
                    retry_until_good=retry_until_good,
                    num_retry_until_good=num_retry_until_good,
                    sample=sample,
                    temperature_increase=temperature_increase,
                    timeout_duration=timeout_duration,
                )
                goods += int(is_good)
                total += 1

                progress.set_description(_description_template(goods, total))
                f.write(output)
        

def main(
        name,
        exist_ok=False,
        *,
        output_dir=SCRIPT_DIR / "outputs",
        secret_path=DEFAULT_PATH_SECRET,
        test_run=False,
        give_model_answers=False,
        timeout_duration=DEFAULT_TIMEOUT_DURATION,
        model=DEFAULT_MODEL_NAME,
        retry_until_good=DEFAULT_RETRY_UNTIL_GOOD,
        num_retry_until_good=DEFAULT_NUM_RETRY_UNTIL_GOOD,
        retry_wait_time=DEFAULT_RETRY_WAIT_TIME,
        temperature_increase=DEFAULT_TEMPERATURE_INCREASE,
    ):
    
    args = locals().copy()
    table = rich.table.Table("Key", "Value", title="Arguments:", show_lines=True)
    for k, v in args.items():
        table.add_row("[bold]" + rich.markup.escape(str(k)), rich.markup.escape(str(v)))
    rich.print(table)

    ###########################################################################
    # Print args
    ###########################################################################
    table = rich.table.Table(
        "Key", 
        "Value", 
        title="[bold blue on white]Arguments:", 
        show_lines=True,
    )
    for k, v in args.items():
        table.add_row(
            "[bold]" + rich.markup.escape(str(k)), 
            rich.markup.escape(str(v)),
        )
    rich.print(table)

    ###########################################################################
    # Check args
    ###########################################################################
    output_dir = pathlib.Path(output_dir)
    assert output_dir.exists(), output_dir
    assert output_dir.is_dir(), output_dir

    # File stuff
    splits_to_paths = {
        split: output_dir / f"{name}.commonsenseqa.chatgpt.{split.value}.jsonl" 
        for split in CVSets
    }

    for path in splits_to_paths.values():
        if not pathvalidate.is_valid_filepath(path.name):
            raise ValueError(
                f"Invalid path: {path}\n"
                f"Fix the value of the name arg: {name}"
            )
        if path.exists() and not exist_ok:
            raise ValueError(f"File exists, and exist_ok=False: {path}")

    ###########################################################################
    # Load OpenAI secret key
    ###########################################################################
    with rich.status.Status("[bold blue on white]Reading secret key."):
        with open(secret_path, "r") as fin:
            openai.api_key = fin.read().strip()


    ###########################################################################
    # Work
    ###########################################################################
    for split in CVSets:
        rich.print(f"[bold blue on white]Running split [green]{split}")
        inference_on_split(
            path=splits_to_paths[split],
            give_model_answers=give_model_answers,
            model=model,
            num_retry_until_good=num_retry_until_good,
            retry_wait_time=retry_wait_time,
            retry_until_good=retry_until_good,
            split=split, 
            temperature_increase=temperature_increase,
            test_run=test_run,
            timeout_duration=timeout_duration,
        )


if __name__ == "__main__":
    fire.Fire(main)

