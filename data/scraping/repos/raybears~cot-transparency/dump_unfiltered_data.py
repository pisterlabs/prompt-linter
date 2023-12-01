from pathlib import Path

from slist import Slist

from cot_transparency.apis.openai import OpenAICompletionPrompt
from cot_transparency.data_models.data import COT_TRAINING_TASKS
from cot_transparency.data_models.io import ExpLoader
from cot_transparency.data_models.models import ExperimentJsonFormat, TaskOutput
from cot_transparency.formatters.core.unbiased import (
    ZeroShotCOTUnbiasedFormatter,
    ZeroShotUnbiasedFormatter,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel


# ruff: noqa: E501


def dump_correct_data(cot_data: bool, exp_dir: str, model: str) -> None:
    jsons = ExpLoader.stage_one(exp_dir=exp_dir)
    for v in jsons.values():
        assert isinstance(v, ExperimentJsonFormat)
    selected_formatter = ZeroShotCOTUnbiasedFormatter.name() if cot_data else ZeroShotUnbiasedFormatter.name()

    # intervention_name should be None
    # dataset should be bbh
    # model should be gpt-3.5-turbo
    tasks = COT_TRAINING_TASKS

    jsons_tasks: Slist[TaskOutput] = (
        Slist(jsons.values())
        .map(lambda x: x.outputs)
        .flatten_list()
        .filter(lambda x: x.task_spec.intervention_name is None)
        .filter(lambda x: x.task_spec.task_name in tasks)
        .filter(lambda x: x.task_spec.inference_config.model == model)
        .filter(lambda x: x.task_spec.formatter_name == selected_formatter)
        # make sure the COTs are distinct
        .distinct_by(
            lambda x: OpenAICompletionPrompt(messages=x.task_spec.messages).format() + x.inference_output.raw_response
        )
    )
    print(f"Number of jsons: {len(jsons_tasks)} for model {model}")
    score = jsons_tasks.map(lambda x: x.is_correct).average()
    print(f"Average score: {score}")

    print(f"Number of jsons: {len(jsons_tasks)}")
    # remove the . from the model name so we are compatible with more file systems
    model_file_name = model.replace(".", "")
    if cot_data:
        write_jsonl_file_from_basemodel(
            path=Path(f"data/training_cots/{model_file_name}_unfiltered.jsonl"),
            basemodels=jsons_tasks,
        )
    else:
        write_jsonl_file_from_basemodel(
            path=Path(f"data/training_non_cots/{model_file_name}_unfiltered.jsonl"),
            basemodels=jsons_tasks,
        )


if __name__ == "__main__":
    dump_correct_data(
        cot_data=True,
        exp_dir="experiments/training_data_1_unfiltered",
        model="gpt-3.5-turbo",
    )
    dump_correct_data(
        cot_data=False,
        exp_dir="experiments/training_data_1_unfiltered",
        model="gpt-3.5-turbo",
    )
