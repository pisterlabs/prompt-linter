import io
import json
import os
import sys
from functools import partial
from types import SimpleNamespace

import evaluate
import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from openai import File, FineTune
from openai.cli import FineTune as cFineTune
from openai.wandb_logger import WandbLogger
from sklearn.metrics import classification_report
from tqdm import tqdm

import wandb

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

PROJECT = "mave"
ENTITY = "parambharat"

default_config = SimpleNamespace(
    model="ada",
    suffix="mave attribute recognition",
    training_file="file-NCUpykAEEMEaMaxXOTeuSf57",
    validation_file="file-LsHbX2sxaHCH9Vhfc0uSycZU",
    n_epochs=1,
    batch_size=4,
    learning_rate_multiplier=0.001,
    prompt_loss_weight=0.1,
    check_if_files_exist=True,
)
run = wandb.init(project=PROJECT, entity=ENTITY, config=default_config)
config = wandb.config


class CustomFineTune(cFineTune):
    @classmethod
    def create(cls, args):
        create_args = {
            "training_file": cls._get_or_upload(
                args.training_file, args.check_if_files_exist
            ),
        }
        if args.validation_file:
            create_args["validation_file"] = cls._get_or_upload(
                args.validation_file, args.check_if_files_exist
            )

        for hparam in (
            "model",
            "suffix",
            "n_epochs",
            "batch_size",
            "learning_rate_multiplier",
            "prompt_loss_weight",
            # "compute_classification_metrics",
            # "classification_n_classes",
            # "classification_positive_class",
            # "classification_betas",
        ):
            attr = getattr(args, hparam)
            if attr is not None:
                create_args[hparam] = attr

        resp = openai.FineTune.create(**create_args)

        # if args.no_follow:
        #     print(resp)
        #     return

        sys.stdout.write(
            "Created fine-tune: {job_id}\n"
            "Streaming events until fine-tuning is complete...\n\n"
            "(Ctrl-C will interrupt the stream, but not cancel the fine-tune)\n".format(
                job_id=resp["id"]
            )
        )
        cls._stream_events(resp["id"])
        return resp


class CustomWandbLogger(WandbLogger):
    @classmethod
    def _log_fine_tune(
        cls,
        fine_tune,
        project,
        entity,
        force,
        show_individual_warnings,
        **kwargs_wandb_init,
    ):
        fine_tune_id = fine_tune.get("id")
        status = fine_tune.get("status")

        # check run completed successfully
        if status != "succeeded":
            if show_individual_warnings:
                print(
                    f'Fine-tune {fine_tune_id} has the status "{status}" and will not be logged'
                )
            return

        # check results are present
        try:
            results_id = fine_tune["result_files"][0]["id"]
            results = File.download(id=results_id).decode("utf-8")
        except:
            if show_individual_warnings:
                print(f"Fine-tune {fine_tune_id} has no results and will not be logged")
            return

        # check run has not been logged already
        run_path = f"{project}/{fine_tune_id}"
        if entity is not None:
            run_path = f"{entity}/{run_path}"
        wandb_run = cls._get_wandb_run(run_path)
        if wandb_run:
            wandb_status = wandb_run.summary.get("status")
            if show_individual_warnings:
                if wandb_status == "succeeded":
                    print(
                        f"Fine-tune {fine_tune_id} has already been logged successfully at {wandb_run.url}"
                    )
                    if not force:
                        print(
                            'Use "--force" in the CLI or "force=True" in python if you want to overwrite previous run'
                        )
                else:
                    print(
                        f"A run for fine-tune {fine_tune_id} was previously created but didn't end successfully"
                    )
                if wandb_status != "succeeded" or force:
                    print(
                        f"A new wandb run will be created for fine-tune {fine_tune_id} and previous run will be overwritten"
                    )
            if wandb_status == "succeeded" and not force:
                return

        # start a wandb run
        wandb.init(
            job_type="fine-tune",
            config=cls._get_config(fine_tune),
            project=project,
            entity=entity,
            name=fine_tune_id,
            **kwargs_wandb_init,
        )

        # log results
        df_results = pd.read_csv(io.StringIO(results))
        for _, row in df_results.iterrows():
            metrics = {k: v for k, v in row.items() if not np.isnan(v)}
            step = metrics.pop("step")
            if step is not None:
                step = int(step)
            wandb.log(metrics, step=step)
        fine_tuned_model = fine_tune.get("fine_tuned_model")
        if fine_tuned_model is not None:
            wandb.summary["fine_tuned_model"] = fine_tuned_model

        # training/validation files and fine-tune details
        cls._log_artifacts(fine_tune, project, entity)

        # mark run as complete
        wandb.summary["status"] = "succeeded"

        return True


def predict_completions(val_df):
    data = []

    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        prompt = row["prompt"]
        res = openai.Completion.create(
            model=wandb.summary["fine_tuned_model"],
            prompt=prompt,
            max_tokens=256,
            stop=["\n\n###\n\n"],
        )
        completion = res["choices"][0]["text"]
        prompt = prompt[:-5]  # remove "\n==>\n"
        target = row["completion"][1:-7]  # remove initial space and "END"
        data.append([prompt, target, completion])
    return pd.DataFrame(data, columns=["prompt", "reference", "prediction"])


def score_dict_similar(row):
    reference = row["reference"]
    prediction = row["prediction"]
    try:
        reference = json.loads(reference)
        prediction = json.loads(prediction)
        common = len(set(reference.items()) & set(prediction.items()))
        actual = len(reference.items())
        return common / actual
    except:
        return 0.0


def prompt_to_bio(row, label_key="target"):
    prompt = row["prompt"]
    target = row[label_key]
    try:
        target = json.loads(target)
        prompt = prompt.split()
        labels = ["O"] * len(prompt)
    except:
        labels = ["O"] * len(prompt)
        return labels

    for attribute, value in target.items():
        values = value.split()
        start_ent = False
        for idx, word in enumerate(values):
            try:
                first_idx = prompt.index(word)
                if idx == 0:
                    first_idx = prompt.index(word)
                    labels[first_idx] = f"B-{attribute}"
                    start_ent = True
                elif start_ent:
                    first_idx = prompt.index(word)
                    labels[first_idx] = f"I-{attribute}"
            except ValueError:
                pass
    return labels


def to_category(row):
    reference = json.loads(row["reference"])["category"]
    try:
        prediction = json.loads(row["prediction"])["category"]
    except:
        return pd.Series({"reference_category": reference, "predicted_category": ""})
    return pd.Series(
        {"reference_category": reference, "predicted_category": prediction}
    )


metric = evaluate.load("seqeval")


def evaluate_results(results_df):
    valid_results_df = results_df[
        results_df.reference_labels.map(len) == results_df.predicted_labels.map(len)
    ]
    if not valid_results_df.empty:
        valid_results_df["exact_match_score"] = valid_results_df.apply(
            score_dict_similar, axis=1
        )
        seq_results = metric.compute(
            predictions=valid_results_df["predicted_labels"].tolist(),
            references=valid_results_df["reference_labels"].tolist(),
        )

        seq_results = (
            pd.DataFrame(seq_results).T.reset_index().rename({"index": "label"}, axis=1)
        )

        clf_results = classification_report(
            y_true=valid_results_df["reference_category"],
            y_pred=valid_results_df["predicted_category"],
            output_dict=True,
        )
        clf_results = (
            pd.DataFrame(clf_results).T.reset_index().rename({"index": "label"}, axis=1)
        )
        return valid_results_df, seq_results, clf_results
    else:
        return results_df, None, None


def train():
    response = CustomFineTune.create(config)
    finetune_id = response["id"]
    finetune = FineTune.retrieve(id=finetune_id)
    CustomWandbLogger._log_fine_tune(
        finetune,
        project=PROJECT,
        entity=ENTITY,
        force=False,
        show_individual_warnings=False,
        id=run._run_id,
        resume="must",
    )
    val_df = pd.read_json(
        "prompts_dataset_val_prepared.jsonl", lines=True, orient="records"
    )
    results_df = predict_completions(val_df)
    prompt_to_labels = partial(prompt_to_bio, label_key="reference")
    prompt_to_predictions = partial(prompt_to_bio, label_key="prediction")
    results_df["reference_labels"] = results_df.apply(prompt_to_labels, axis=1)
    results_df["predicted_labels"] = results_df.apply(prompt_to_predictions, axis=1)
    results_df[["reference_category", "predicted_category"]] = results_df.apply(
        to_category, axis=1
    )

    results_df, seq_results, clf_results = evaluate_results(results_df)
    if seq_results is not None and clf_results is not None:
        wandb.log(
            {
                "eval_predictions": wandb.Table(dataframe=results_df),
                "seq_eval_metrics": wandb.Table(dataframe=seq_results),
                "classification_metrics": wandb.Table(dataframe=clf_results),
            }
        )
        wandb.log({"exact_match_score": results_df["exact_match_score"].mean()})
        wandb.run.summary[
            "exact_match_metrics"
        ] = results_df.exact_match_score.describe().to_dict()
        wandb.run.summary["classification_metrics"] = json.loads(
            clf_results.loc[22:].set_index("label").T.to_json()
        )
        wandb.run.summary["seqeval_metrics"] = json.loads(
            seq_results[(seq_results.label.str.startswith("overall"))]
            .set_index("label")
            .T.to_json()
        )
    else:
        wandb.log(
            {"eval_predictions": wandb.Table(dataframe=results_df),}
        )
        wandb.log({"exact_match_score": results_df["exact_match_score"].mean()})
    wandb.finish()


if __name__ == "__main__":
    train()
