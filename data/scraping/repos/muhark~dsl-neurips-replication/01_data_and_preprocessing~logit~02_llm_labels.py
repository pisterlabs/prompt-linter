#! /usr/bin/python

"""
@title LLM Label Pipeline

Quick and script for doing zero/few-shot classification with fixed exemplars.

Usage example:

```
python 02_llm_labels.py \
    --dataset_name_or_path cbp_data/cbp_data_no_labels.csv \
    --model_args llm_configs/model_args.json \
    --prompt_template llm_configs/cbp_binary_macro_0shot.txt \
    --testing
```
"""
from __future__ import annotations

import os
import json
import math
import argparse
import logging
from configparser import ConfigParser
from pathlib import Path
from typing import List, Dict
from datetime import datetime as dt

import openai
import datasets
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Logger init
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger()


# Set up args
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="In-context classification using OpenAI API."
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default="llm_configs/model_args.json",
        help=(
            "Path to OpenAI model arguements in JSON format. Defaults to `model_args.json`."
        )
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default=None,
        help=(
            "Name of dataset provided by `datasets` to use for classification or path to local data files (csv json supported)."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cbp_data",
        help=("Name of directory to save results")
    )
    parser.add_argument(
        "--feature_col",
        type=str,
        default="text",
        help=(
            "Name of column corresponding to feature to be provided to in-context learner."
        )
    )
    parser.add_argument(
        "--local_dataset",
        type=bool,
        default=True,
        help=("Flag that dataset is local and should not be pulled from HF")
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="llm_configs/prompt.txt",
        help=("Path to prompt template. Defaults to `prompt.txt`.")
    )
    parser.add_argument(
        "--tolerance", type=int, default=100, help=("Number of failed results before ")
    )
    parser.add_argument(
        "--api_cfg_dir",
        type=str,
        default="~/.cfg/openai.cfg",
        help=("Provide config path for credentials options")
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help=("Name of directory to save results")
    )
    parser.add_argument(
        "--disable_tqdm",
        type=bool,
        default=False,
        help=("Disable use of tqdm progress bars.")
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help=("Flag for test mode. Checks with single observation.")
    )
    args = parser.parse_args()
    return args


# Setup OpenAI Key
def seed_openai_key(cfg: str = "~/.cfg/openai.cfg") -> str:
    """
    Reads OpenAI key from config file and adds it to environment.
    Assumed config location is "~/.cfg/openai.cfg"
    """
    # Get OpenAI Key
    config = ConfigParser()
    try:
        config.read(Path(cfg).expanduser())
    except:
        raise ValueError(
            f"Could not using read file at config_dir: {cfg}. Please contact me on how to set up your OpenAI config."
        )
    openai_key = config["API_KEY"]["secret"]
    openai.api_key = openai_key
    os.environ["OPENAI_API_KEY"] = openai_key
    return openai_key


def get_prompt_template(prompt_file: str | Path) -> PromptTemplate:
    if isinstance(prompt_file, str):
        prompt_file = Path(prompt_file)
    template = PromptTemplate(
        template=prompt_file.read_text(), input_variables=["content"]
    )
    return template


# Load data
def load_data(dataset_file: str, local_dataset: bool) -> datasets.Dataset:
    "Dataset loading logic. Currently returns single split ('train')."
    if local_dataset:
        extension = dataset_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        if extension == "jsonl":
            extension = "json"
        data_files = {"train": dataset_file}
        ds = load_dataset(extension, data_files=data_files)["train"]
    else:
        ds = load_dataset(dataset_file)["train"]
    return ds


# openai API direct with exponential backoff
@retry(wait=wait_random_exponential(min=1, max=66), stop=stop_after_attempt(6))
def call_openai(prompt: str, kwargs):
    return openai.Completion.create(prompt=prompt, **kwargs)


def construct_dataframe(
    ds: datasets.Dataset, results: List[Dict], col_ref: str, n_logprobs: int = 5
) -> pd.DataFrame:
    """
    Function for converting output to formatted dataframe.
    `n_logprobs` controls the number of logprobs to save. Should be inherited from `model_args`, but can be varied for debugging etc.
    """
    df = ds.to_pandas()
    df[f"q_{col_ref}"] = [result["choices"][0]["text"] for result in results]

    res_cols = [[] for i in range(5)]
    res_prob_cols = [[] for i in range(5)]
    for result in results:
        top_logprobs = result["choices"][0]["logprobs"]["top_logprobs"][0]
        for i in range(5):
            res, res_logprob = list(top_logprobs.items())[i]
            res_prob = math.exp(res_logprob)
            res_cols[i].append(res)
            res_prob_cols[i].append(res_prob)
    for i in range(5):
        df[f"res{i+1}_{col_ref}"] = res_cols[i]
        df[f"res{i+1}_prob_{col_ref}"] = res_prob_cols[i]
    return df


def classification_loop(
    ds: datasets.Dataset,
    model_args: Dict,
    template: PromptTemplate,
    feature_col: str = "text",
    max_loops: int = -1,
    time_now: str = dt.now().strftime("%Y%m%d-%H%M%S"),
    disable_tqdm: bool = False,
    logs_dir: Path = Path("logs/"),
) -> List[Dict]:
    # logs is specifically for dealing with failures etc.
    outfile = logs_dir / f"run-{time_now}.json"
    results = []
    try:
        for i, row in enumerate(tqdm(ds, disable=disable_tqdm)):
            prompt = template.format(content=row[feature_col].strip())
            result = call_openai(prompt, model_args)
            results.append(result)
            if (i + 1) == max_loops:
                break
    except:  # Naked execpt, blame the OpenAI API
        outfile.write_text(json.dumps(results))
        return results
    return results


def main():
    # Get args
    args = parse_args()
    model_args = json.loads(Path(args.model_args).read_text())
    model_ref = model_args["engine"]

    # Create dirs
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    for dir in [log_dir, output_dir]:
        if not dir.exists():
            logging.info(f"Creating {dir}")
            dir.mkdir(parents=True)

    # Seed openai
    seed_openai_key(args.api_cfg_dir)
    time_now = dt.now().strftime("%Y%m%d-%H%M%S")

    # Load dataset
    ds = load_data(args.dataset_name_or_path, local_dataset=args.local_dataset)
    logger.info(f"Loaded dataset. N observations: {len(ds):,}")
    if args.testing:
        logger.info("Running in test mode with 2 observations.")
        ds = ds.select(range(2))

    # Run classifications
    logger.info("Starting classification loop...")
    results = classification_loop(
        ds=ds,
        model_args=model_args,
        template=get_prompt_template(prompt_file=args.prompt_template),
        feature_col=args.feature_col,
        max_loops=-1,
        time_now=time_now,
        disable_tqdm=args.disable_tqdm
    )
    if not results:  # This shouldn't happen but it did so I'll leave it
        raise Exception("Empty results, please check the log.")

    # Create dataframe
    prompt_name = Path(args.prompt_template).stem
    dataset_name = Path(args.dataset_name_or_path).stem

    try:
        df = construct_dataframe(
            ds=ds,
            results=results,
            col_ref=model_ref + "_" + prompt_name.split("_")[-1],
            n_logprobs=model_args.get("logprobs"))
    except:
        pass
    finally:
        Path(
            log_dir / f"{dataset_name}_{model_ref}_{prompt_name}.json"
        ).write_text(json.dumps(results))

    # Finally, save output
    if args.testing:
        logger.info(f"Output\n{df.to_string()}")
        df.to_csv(
            output_dir / f"{dataset_name}_{model_ref}_{prompt_name}_test.csv", index=False)
    else:
        df.to_csv(
            output_dir / f"{dataset_name}_{model_ref}_{prompt_name}.csv", index=False)


if __name__ == "__main__":
    main()
