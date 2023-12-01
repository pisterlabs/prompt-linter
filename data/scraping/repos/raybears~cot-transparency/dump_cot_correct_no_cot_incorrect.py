from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from slist import Slist

from cot_transparency.apis.openai import OpenAICompletionPrompt
from cot_transparency.data_models.data import COT_TESTING_TASKS
from cot_transparency.data_models.io import ExpLoader
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.json_utils.read_write import (
    write_csv_file_from_basemodel,
    write_jsonl_file_from_basemodel,
)


# ruff: noqa: E501


class FlatCOTComparisonOutput(BaseModel):
    cot_prompt: str
    no_cot_prompt: str
    ground_truth: str
    cot_full_response: str
    no_cot_full_response: str
    cot_parsed_answer: str
    no_cot_parsed_answer: str
    task_name: str


def to_flat(tup: tuple[TaskOutput, TaskOutput]) -> Optional[FlatCOTComparisonOutput]:
    cot = tup[0]
    no_cot = tup[1]
    if cot.is_correct and not no_cot.is_correct:
        return FlatCOTComparisonOutput(
            cot_prompt=OpenAICompletionPrompt(messages=cot.task_spec.messages).format(),
            no_cot_prompt=OpenAICompletionPrompt(messages=no_cot.task_spec.messages).format(),
            ground_truth=cot.task_spec.ground_truth,
            cot_full_response=cot.inference_output.raw_response,
            no_cot_full_response=no_cot.inference_output.raw_response,
            cot_parsed_answer=cot.inference_output.parsed_response,  # type: ignore
            no_cot_parsed_answer=no_cot.inference_output.parsed_response,  # type: ignore
            task_name=cot.task_spec.task_name,
        )
    else:
        return None


def dump_cot_comparison():
    """Produces a dataset containing instances where the model was wrong without COT, but correct with COT
    Steps
    1. Run stage one with an unbiased formatter e.g.
     python stage_one.py --exp_dir experiments/gpt_35_cot --models "['gpt-3.5-turbo']" --formatters "['ZeroShotCOTUnbiasedFormatter', 'ZeroShotUnbiasedFormatter']" --dataset cot_testing
    2. Run this script
    3. This will produce a csv file
    """
    jsons = ExpLoader.stage_one("experiments/gpt_35_cot")
    model: str = "gpt-3.5-turbo"
    selected_formatters = {"ZeroShotCOTUnbiasedFormatter", "ZeroShotUnbiasedFormatter"}

    tasks = COT_TESTING_TASKS

    jsons_tasks: Slist[TaskOutput] = (
        Slist(jsons.values())
        .map(lambda x: x.outputs)
        .flatten_list()
        .filter(lambda x: x.task_spec.formatter_name in selected_formatters)
        .filter(lambda x: x.task_spec.intervention_name is None)
        .filter(lambda x: x.task_spec.task_name in tasks)
        .filter(lambda x: x.task_spec.inference_config.model == model)
    )
    cot, no_cot = jsons_tasks.split_by(lambda x: x.task_spec.formatter_name == "ZeroShotCOTUnbiasedFormatter")
    print(f"len(cot): {len(cot)}")
    print(f"len(no_cot): {len(no_cot)}")
    # make a make of no_cot
    no_cot_map: dict[str, TaskOutput] = {x.task_spec.task_hash: x for x in no_cot}
    cot_tups = cot.map(
        lambda x: (x, no_cot_map[x.task_spec.task_hash]) if x.task_spec.task_hash in no_cot_map else None
    ).flatten_option()
    print(f"len(cot_tups): {len(cot_tups)}")
    flat_cot_tups: Slist[FlatCOTComparisonOutput] = cot_tups.map(to_flat).flatten_option()
    print(f"len(flat_cot_tups): {len(flat_cot_tups)}")
    write_csv_file_from_basemodel(
        path=Path("cot_comparison.csv"),
        basemodels=flat_cot_tups,
    )
    write_jsonl_file_from_basemodel(
        path=Path("cot_comparison.jsonl"),
        basemodels=flat_cot_tups,
    )


if __name__ == "__main__":
    dump_cot_comparison()
