from collections import defaultdict
from datetime import datetime
import importlib
import logging
import hydra
import os

from src.helpers import (
    save_results_to_file,
    MultiMetric,
    chunks,
    log_config,
    log_prompt_templates,
    log_all_results,
    set_seed,
)
from src.models import CohereModelWrapper, OpenAIModel
from src.dataset import ParticularisedImplicatureDataset
from src.prompting import read_prompt_file
from src.task import RankingTask, Task, CompletionTask
import pandas as pd

logger = logging.getLogger()


def get_logging(arguments):
    log_config(arguments)
    set_seed(arguments.seed)

    # Make results folder if it doesn't exist yet.
    if not os.path.exists(os.path.join(os.getcwd(), arguments.results_folder)):
        os.mkdir(os.path.join(os.getcwd(), arguments.results_folder))


def load_datasets(arguments):
    dataset = ParticularisedImplicatureDataset.read_data_csv(
        arguments.test_input_data_path, arguments.dev_input_data_path, arguments.seed
    )
    return dataset


def load_models(arguments, prompt_variations):
    model_ids = arguments.model_ids.split(";")
    models = []
    for model_id in model_ids:
        if "cohere" in model_id:
            size = model_id.split("-")[-1]
            model = CohereModelWrapper(model_size=size)
            assert (
                arguments.task == "ranking" or arguments.task == "completion"
            ), "Only ranking task implemented for Cohere models."
        elif "openai" in model_id:
            engine = model_id.split("-")[-1]
            model = OpenAIModel(model_engine=engine, rate_limit=False)
        else:
            raise ValueError("Unrecognized model_id=%s" % model_id)

        # Prepare metrics.
        implicature_metrics = MultiMetric(num_metrics=len(prompt_variations))
        completion_metrics = MultiMetric(num_metrics=len(prompt_variations))
        bigger_is_better = True if arguments.task == "completion" else False
        if arguments.task == "completion":
            assert "openai" in arguments.model_ids or "cohere" in arguments.model_ids, \
                "Only OpenAI and Cohere models implemented for completion task."

        models.append(
            {
                "model": model,
                "model_id": model_id,
                "implicature_metrics": implicature_metrics,
                "completion_metrics": completion_metrics,
                "bigger_is_better": bigger_is_better,
            }
        )
    return models


def get_prompts(arguments):
    # Prepare prompts.
    prompt_variations = read_prompt_file(arguments.prompt_file)
    module = importlib.import_module("src.prompting")
    prompt_class = getattr(module, arguments.prompt_class)
    prompt_templates = [
        prompt_class(
            **prompt_variation,
            contrastive=True if arguments.task == "contrastive" else False,
        )
        for prompt_variation in prompt_variations
    ]
    return prompt_variations, prompt_templates


def get_tasks(arguments):
    if arguments.task == "ranking":
        task_handler = RankingTask()
    elif arguments.task == "contrastive":
        # The contrastive task is only different in the prompt
        task_handler = RankingTask()
    elif arguments.task == "completion":
        task_handler = CompletionTask()
    else:
        raise ValueError(f"Unknown --task {arguments.task}")
    return task_handler


def dataloader(
    dataset,
    prompt_templates,
    task_handler: Task,
    arguments,
):

    datapoint_idx = 0
    for csv_idx, implicature in enumerate(
        dataset.get_implicature_iterator(k_shot=arguments.k_shot)
    ):
        # historically iterate this way
        prepared_example = task_handler.prepare_for_task(implicature, arguments.random_labels)

        for prompt_idx, prompt_template in enumerate(prompt_templates):
            _example = implicature
            if arguments.task == "ranking" or arguments.task == "contrastive":
                for label_idx, label_type in enumerate(
                    ["correct_example", "false_example"]
                ):
                    prompt = prepared_example["prompt_examples"]
                    label = prepared_example[label_type]
                    is_false_example = True if label_type == "false_example" else False
                    label = task_handler.prepare_datapoint(
                        label, prompt_template, is_false_example, prompt
                    )
                    info = {"example": _example, "prompt": prompt}
                    yield csv_idx, prompt_idx, label_idx, datapoint_idx, label, info
                    datapoint_idx += 1
            elif arguments.task == "completion":
                prompt = prepared_example["prompt_examples"]
                label = prepared_example["test_example"]
                is_false_example = False
                label = task_handler.prepare_datapoint(
                    label, prompt_template, is_false_example, prompt
                )
                info = {"example": _example, "prompt": prompt}
                yield csv_idx, prompt_idx, 0, datapoint_idx, label, info
                datapoint_idx += 1
            else:
                raise ValueError(f"Unknown --task {arguments.task}")


def save_results(
    results,
    old_df,
    example_result,
    correct_per_variation,
    valid_completions,
    all_results,
    implicature_write_lines,
    prompt_variations,
    prompt_templates,
    models,
    write_data_to,
    write_results_to,
    arguments,
):
    old_num_rows = old_df.shape[0]
    df = pd.DataFrame(results)
    df.to_csv("raw.csv")

    pos_df = df[df.label_idx == 0].drop(columns="label_idx").reset_index()
    neg_df = df[df.label_idx == 1].drop(columns="label_idx").reset_index()
    neg_df.rename(columns=lambda x: "negative_" + x, inplace=True)
    new_df = pos_df.join(neg_df)

    impl_df = new_df
    new_new_df = impl_df
    if arguments.task == "ranking" or arguments.task == "contrastive":
        new_new_df["task_is_correct"] = new_new_df["score"] < new_new_df["negative_score"]
        new_new_df["valid_completion"] = [0] * len(new_new_df["task_is_correct"])
    elif arguments.task == "completion":
        task_is_correct_list = []
        valid_completion_list = []
        for i, example in enumerate(new_new_df.example):
            completion = new_new_df["score"][i]
            split_answer = completion.lower().split("answer:")[-1]
            true_answer = example["example"]["implicature"]
            # If completion output was not specified properly, assume answer is last word of output.
            if "answer:" not in completion.lower():
                split_answer = split_answer[-10:]
            if "no" not in split_answer.lower() and "yes" not in split_answer.lower():
                valid_completion_ex = False
            else:
                valid_completion_ex = True
            if true_answer.lower() in split_answer.lower():
                task_correct = True
            else:
                task_correct = False
            task_is_correct_list.append(task_correct)
            valid_completion_list.append(valid_completion_ex)
        new_new_df["task_is_correct"] = task_is_correct_list
        new_new_df["valid_completion"] = valid_completion_list
    else:
        raise ValueError(f"Unknown --task {arguments.task}")

    new_new_df.to_csv("akbir_test.csv")

    new_new_df = pd.concat((old_df, new_new_df))

    for row in new_new_df[old_num_rows:].iterrows():
        _, row = row
        model_idx = row["model_idx"]
        model_d = models[model_idx]
        prompt_idx = row["prompt_idx"]
        task_is_correct = row["task_is_correct"]
        valid_completion = row["valid_completion"]

        # regular method for adding things to loggers
        example_correct = model_d["implicature_metrics"].update(
            index_to_update=prompt_idx, correct=task_is_correct
        )
        valid_completion = model_d["completion_metrics"].update(
            index_to_update=prompt_idx, correct=valid_completion
        )

        result = {
            "correct_score": row["score"],
            "false_score": row["negative_score"],
            "scored_texts": {
                "correct": row["label"],
                "false": row["negative_label"],
            },
        }
        result["task_correct"] = example_correct
        result["valid_completion"] = valid_completion
        correct_per_variation += result["task_correct"]
        valid_completions += result["valid_completion"]

        example_result[model_d["model_id"]][f"prompt_template_{prompt_idx}"] = {
            "implicature_result": {
                "example_correct": result["task_correct"],
                "valid_completion": result["valid_completion"],
                "correct_score": row["score"],
                "false_score": row["negative_score"],
                "scored_texts": result["scored_texts"],
            }
        }

        if model_idx == 0:  # Only write once for all models
            implicature_write_lines.append(tuple(result["scored_texts"].values()))

        if prompt_idx == len(prompt_templates) - 1:
            # finished all prompt variants
            example_result[model_d["model_id"]]["average_implicature_accuracy"] = (
                correct_per_variation / len(prompt_variations)
            ) * 100.0

            example_result[model_d["model_id"]]["average_valid_completion"] = (
                valid_completions / len(prompt_variations)
            ) * 100.0

            example_result["original_example"] = row["example"]["example"]
            example_result["prompt_examples"] = row["prompt"]
            all_results.append(example_result)

            # reset example specific metrics
            example_result = defaultdict(lambda: defaultdict())
            correct_per_variation = 0
            valid_completions = 0
    save_results_to_file(
        len(prompt_templates),
        models,
        all_results,
        implicature_write_lines,
        write_data_to,
        write_results_to,
        arguments,
    )
    return (
        new_new_df,
        example_result,
        correct_per_variation,
        valid_completions,
        write_data_to,
        write_results_to,
    )


@hydra.main(config_path="configs", config_name="config")
def main(arguments):
    get_logging(arguments)
    dataset = load_datasets(arguments)

    # Prepare prompts
    prompt_variations, prompt_templates = get_prompts(arguments)
    log_prompt_templates(prompt_templates, k_shot=arguments.k_shot)

    # Load models and metrics for each.
    models = load_models(arguments, prompt_variations)

    # Get task object
    task_handler = get_tasks(arguments)

    # Lists for keeping track of results and data, will be written to results folder.
    results = []
    df = pd.DataFrame(results)
    all_results = []
    implicature_write_lines = []
    example_result = defaultdict(lambda: defaultdict())
    correct_per_variation = 0
    valid_completions = 0

    # Path to save all results (intermediate and final)
    save_file_data = f"data_{arguments.task}_task_{arguments.k_shot}_nprompts_{len(prompt_templates)}_npromptvars_{str(datetime.today())}.json"
    write_data_to = os.path.join(arguments.results_folder, save_file_data)
    save_file_results = f"results_{arguments.task}_task_{arguments.k_shot}_nprompts_{len(prompt_templates)}_npromptvars_{len(models)}_models_{str(datetime.today())}.json"
    write_results_to = os.path.join(arguments.results_folder, save_file_results)

    # Loop over the models that are being evaluated.
    for _model_idx, model_d in enumerate(models):
        logger.info(f"Processing Model: {model_d['model_id']}")
        model_d["model"].to(arguments.device)
        set_seed(arguments.seed)
        dataset_iterator = list(
            dataloader(
                dataset,
                prompt_templates,
                task_handler,
                arguments,
            )
        )
        # query the LLMs
        for (
            _csv_idx,
            _prompt_idx,
            _label_idx,
            _datapoint_idx,
            _label,
            _info,
        ) in chunks(dataset_iterator, arguments.batch_size):
            if (
                arguments.max_num_evaluations != -1
                and max(_csv_idx) >= arguments.max_num_evaluations
            ):
                logger.info(
                    "Hit max num evaluations %d." % arguments.max_num_evaluations
                )
                break

            if (
                _datapoint_idx[-1] + 1
            ) % arguments.logging_frequency == 0 and _datapoint_idx[-1] > 0:
                logger.info(
                    "Processed %d/%d datapoints."
                    % (_datapoint_idx[-1], len(dataset_iterator))
                )

            if (
                arguments.skip_until_idx != -1
                and max(_csv_idx) < arguments.skip_until_idx
            ):
                continue
            # Get log probabilities for the damm datapoint
            if arguments.task != "completion":
                _scores = model_d["model"].get_model_score(_label)
            else:
                _scores = model_d["model"].get_model_completion(_label)
            batch_results = []
            for batch_idx, score in enumerate(_scores):
                results.append(
                    {
                        "model_idx": _model_idx,
                        "csv_idx": _csv_idx[batch_idx],
                        "prompt_idx": _prompt_idx[batch_idx],
                        "label_idx": _label_idx[batch_idx],
                        "score": score,
                        "label": _label[batch_idx],
                    }
                    | _info[batch_idx]
                )
                batch_results.append(
                    {
                        "model_idx": _model_idx,
                        "csv_idx": _csv_idx[batch_idx],
                        "prompt_idx": _prompt_idx[batch_idx],
                        "label_idx": _label_idx[batch_idx],
                        "score": score,
                        "label": _label[batch_idx],
                    }
                    | _info[batch_idx]
                )
            df, example_result, correct_per_variation, valid_completions, _, _ = save_results(
                batch_results,
                df,
                example_result,
                correct_per_variation,
                valid_completions,
                all_results,
                implicature_write_lines,
                prompt_variations,
                prompt_templates,
                models,
                write_data_to=write_data_to,
                write_results_to=write_results_to,
                arguments=arguments,
            )
        # put model back onto cpu
        model_d["model"].to("cpu")

    log_all_results(models)

    logger.info(f"Wrote data to: {write_data_to}")
    logger.info(f"Wrote results to: {write_results_to}")


if __name__ == "__main__":
    main()
