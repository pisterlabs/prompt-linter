from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from slist import Slist

from cot_transparency.apis.openai import OpenAICompletionPrompt
from cot_transparency.data_models.data import COT_TESTING_TASKS
from cot_transparency.data_models.data.bbh import MilesBBHRawData
from cot_transparency.data_models.data.bbh_biased_wrong_cot import BiasedWrongCOTBBH
from cot_transparency.data_models.io import ExpLoader
from cot_transparency.data_models.models import ExperimentJsonFormat, TaskOutput
from cot_transparency.formatters.extraction import BREAK_WORDS
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
)
from cot_transparency.json_utils.read_write import write_csv_file_from_basemodel


# ruff: noqa: E501


class FlatSimple(BaseModel):
    biased_prompt: str
    task: str
    biased_formatter: str
    biased_context_response: str
    debiased_context_response: str
    debiased_prompt: str
    biased_context_parsed_response: str | None
    ground_truth: str
    biased_on_ans: str


def cot_extraction(completion: str) -> Optional[str]:
    """Extracts the biased cot from the completion
    This is done by taking the lines up til the first that contains best answer is: (
    """
    lines = completion.split("\n")
    line_no: Optional[int] = None
    for idx, line in enumerate(lines):
        for break_word in BREAK_WORDS:
            if break_word in line:
                line_no = idx
                break
    # join the lines up til the line that contains best answer is: (
    return "\n".join(lines[:line_no]) if line_no is not None else None


def task_output_to_bad_cot(task: TaskOutput) -> Optional[BiasedWrongCOTBBH]:
    # extract out the bad cot
    bad_cot = cot_extraction(task.first_raw_response)
    raw_data = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    return (
        BiasedWrongCOTBBH(
            idx=raw_data.idx,
            inputs=raw_data.inputs,
            targets=raw_data.targets,
            multiple_choice_targets=raw_data.multiple_choice_targets,
            multiple_choice_scores=raw_data.multiple_choice_scores,
            split=raw_data.split,
            random_ans_idx=raw_data.random_ans_idx,
            parsed_inputs=raw_data.parsed_inputs,
            cot=bad_cot,
            task=task.task_spec.task_name,
        )
        if bad_cot is not None
        else None
    )


def task_output_to_flat(biased_task: TaskOutput, unbiased_task: TaskOutput) -> FlatSimple:
    converted = OpenAICompletionPrompt(messages=biased_task.task_spec.messages).format()
    return FlatSimple(
        biased_prompt=converted,
        biased_context_response=biased_task.first_raw_response,
        ground_truth=biased_task.task_spec.ground_truth,
        biased_on_ans=biased_task.task_spec.biased_ans,  # type: ignore
        biased_formatter=biased_task.task_spec.formatter_name,
        biased_context_parsed_response=biased_task.first_parsed_response,
        task=biased_task.task_spec.task_name,
        debiased_context_response=unbiased_task.first_raw_response,
        debiased_prompt=OpenAICompletionPrompt(messages=unbiased_task.task_spec.messages).format(),
    )


def filter_for_biased_wrong(jsons_tasks: Slist[TaskOutput]) -> Slist[TaskOutput]:
    results: Slist[TaskOutput] = (
        # only get the ones that are biased
        jsons_tasks.filter(lambda x: x.task_spec.biased_ans == x.first_parsed_response)
        # Sometimes we have multiple runs of the same task, we want to get the first one
        .distinct_by(
            lambda x: x.task_spec.task_name
            + x.task_spec.task_hash
            + x.task_spec.inference_config.model_hash()
            + x.task_spec.formatter_name
        )
        # only get the ones that are wrong
        .filter(lambda x: x.task_spec.biased_ans != x.task_spec.ground_truth)
    )
    return results


if __name__ == "__main__":
    """Produces a dataset containing answers that
    - are biased towards the user's choice
    - are wrong
    Steps
    1. Run stage one with a biased formatter
     `python stage_one.py --exp_dir experiments/bad_cot --models '["gpt-3.5-turbo"]' --formatters '["ZeroShotCOTSycophancyFormatter"]'`
    2. Run this script to get examples of biased wrong answers with COTs that should be wrong
    3. This will produce a data.jsonl file in data/bbh_biased_wrong_cot
    4. Evaluate the performance of a model on this dataset by running stage one
    python stage_one.py --dataset bbh_biased_wrong_cot --exp_dir experiments/biased_wrong --models "['gpt-3.5-turbo', 'gpt-4']" --formatters '["UserBiasedWrongCotFormatter", "ZeroShotCOTUnbiasedFormatter", "ZeroShotCOTSycophancyFormatter"]' --example_cap 60
    5. Run the following to get the overall accuracy
    python analysis.py accuracy experiments/biased_wrong
    """
    jsons = ExpLoader.stage_one("experiments/finetune")
    model = "gpt-3.5-turbo"
    for v in jsons.values():
        assert isinstance(v, ExperimentJsonFormat)

    jsons_tasks: Slist[TaskOutput] = Slist(jsons.values()).map(lambda x: x.outputs).flatten_list()  # type: ignore
    selected_formatters = Slist(
        [
            WrongFewShotIgnoreMistakesBiasedFormatter,
            # MoreRewardBiasedFormatter,
            # ZeroShotCOTSycophancyFormatter,
            # StanfordBiasedFormatter,
        ]
    ).map(lambda x: x.name())
    tasks = COT_TESTING_TASKS
    intervention = None
    print(f"Number of jsons: {len(jsons_tasks)}")

    filtered_for_formatters = (
        jsons_tasks.filter(lambda x: x.task_spec.formatter_name in selected_formatters)
        .filter(lambda x: x.task_spec.intervention_name == intervention)
        .filter(lambda x: x.task_spec.task_name in tasks)
        .filter(lambda x: x.task_spec.inference_config.model == model)
    )
    results: Slist[TaskOutput] = filter_for_biased_wrong(filtered_for_formatters)

    debiased_formatter = WrongFewShotIgnoreMistakesBiasedFormatter.name()
    debiased_results: Slist[TaskOutput] = jsons_tasks.filter(
        lambda x: x.task_spec.formatter_name == debiased_formatter
        and x.task_spec.intervention_name is None
        and x.task_spec.inference_config.model == "ft:gpt-3.5-turbo-0613:academicsnyuperez::7tWKhqqg"
    ).filter(lambda x: x.is_correct)
    unbiased_hash: dict[str, TaskOutput] = debiased_results.map(lambda x: (x.task_spec.task_hash, x)).to_dict()
    joined: Slist[tuple[TaskOutput, TaskOutput]] = results.map(
        lambda x: (x, unbiased_hash[x.task_spec.task_hash]) if x.task_spec.task_hash in unbiased_hash else None
    ).flatten_option()
    print(f"Number of joined: {len(joined)}")

    # convert to MilesBBHWithBadCot
    # converted: Slist[BiasedWrongCOTBBH] = results.map(task_output_to_bad_cot).flatten_option()
    # write to jsonl
    # write_jsonl_file_from_basemodel(path=Path("data/bbh_biased_wrong_cot/data.jsonl"), basemodels=converted)

    # This is if you want to view them as a CSV
    flattened: Slist[FlatSimple] = joined.map(lambda x: task_output_to_flat(x[0], x[1]))
    print(f"Number of flattened: {len(flattened)}")
    write_csv_file_from_basemodel(path=Path(f"miles_{model}.csv"), basemodels=flattened)
