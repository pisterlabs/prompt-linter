import os
import openai
import evaluate
import wandb
import pandas as pd
import numpy as np

PROJECT_NAME = "semplify"
default_config = dict(
    model_name="text-davinci-002",
    temperature=1,
    max_tokens=80,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop_seq="###",
    num_generations=1,
)
wandb.init(project=PROJECT_NAME, config=default_config)
config = wandb.config

cleaned_artifacts = wandb.use_artifact("processed_dataset:latest")
cleaned_artifacts = cleaned_artifacts.wait()

zeroshot_table = cleaned_artifacts.get("zeroshot_dataset")
zeroshot_df = pd.DataFrame(zeroshot_table.data, columns=zeroshot_table.columns)

openai.api_key = os.getenv("OPENAI_API_KEY")
bleu = evaluate.load("google_bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


def run_inference(prompt):
    response = openai.Completion.create(
        model=config.model_name,
        prompt=prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty,
        stop=config.stop_seq,
        n=config.num_generations,
    )

    generated_tweets = [
        choice.to_dict()["text"].strip() for choice in response.to_dict()["choices"]
    ]
    return generated_tweets


def evaluate_inference(references, predictions):
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions, str):
        predictions = [predictions]
    if len(references) == 1 and len(predictions) > 1:
        references = references * len(predictions)

    bleu_results = bleu.compute(predictions=predictions, references=references)[
        "google_bleu"
    ]
    rouge_results = rouge.compute(predictions=predictions, references=references)[
        "rougeLsum"
    ]
    bertscore_results = bertscore.compute(
        predictions=predictions, references=references, lang="en"
    )["f1"][0]
    final_results = {
        "bleu": bleu_results,
        "rogue": rouge_results,
        "bert_f1": bertscore_results,
    }
    final_results["mean_score"] = float(np.average(list(final_results.values())))
    return pd.Series(final_results)


def main():
    df = zeroshot_df.copy()
    df["predictions"] = df["prompt"].map(run_inference)
    df[["bleu", "rogue", "bert_f1", "mean_score"]] = df.apply(
        lambda x: evaluate_inference(
            predictions=x["predictions"], references=x["tweet"]
        ),
        axis=1,
    )
    wandb.log({"mean_score": df["mean_score"].mean()})
    predictions_table = wandb.Table(dataframe=df)
    predictions_artifact = wandb.Artifact("predictions", type="dataset")
    predictions_artifact.add(
        predictions_table, f"zeroshot_predictions-{wandb.run.name}"
    )
    wandb.log_artifact(predictions_artifact)
    wandb.log({"zeroshot_predictions": predictions_table})

    return df


if __name__ == "__main__":
    main()
