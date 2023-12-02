import os
import openai
import evaluate
import wandb
import pandas as pd
import numpy as np
from transformers import GPT2TokenizerFast

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

cleaned_table = cleaned_artifacts.get("cleaned_data")
cleaned_df = pd.DataFrame(cleaned_table.data, columns=cleaned_table.columns)

openai.api_key = os.getenv("OPENAI_API_KEY")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
bleu = evaluate.load("google_bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


def make_history(df, prompt_length=2000, shuffled=False):
    if shuffled:
        df = df.sample(frac=1, replace=False)
    total_length = 0
    prompts = []
    for idx, item in df.iloc[::-1].iterrows():
        if item["prompt_length"] + total_length < prompt_length:
            total_length += item["prompt_length"]
            row_prompt = (
                "[changelog]:\n\n"
                + item["cleaned_logs"].strip()
                + "\n\n[tweet]:\n\n"
                + item["cleaned_tweet"].strip()
                + "\n\n###\n\n"
            )
            prompts.append(row_prompt)

    return "\n".join(prompts[::-1])


def make_few_shot_dataset(df, prompt_length, shuffled):
    prompts = []
    tweets = []
    for idx in range(1, len(df)):
        current_row = df.iloc[idx]
        previous_rows = df.iloc[:idx]
        history_length = prompt_length - current_row["log_length"] - 10
        history_prompt = make_history(previous_rows, history_length, shuffled=shuffled)
        prompt = (
            history_prompt
            + "[changelog]:\n\n"
            + current_row["cleaned_logs"].strip()
            + "\n\n[tweet]:\n\n"
        )
        prompts.append(prompt)
        tweets.append(current_row["cleaned_tweet"].strip())
    return pd.DataFrame(
        {"prompt": prompts, "reference": tweets}, index=range(1, len(df))
    )


def get_tokenized_len(texts):
    return list(map(len, tokenizer(texts).input_ids))


def load_experiment_dataset(df):
    df["tweet_length"] = get_tokenized_len(df["cleaned_tweet"].tolist())
    df["log_length"] = get_tokenized_len(df["cleaned_logs"].tolist())
    df["prompt_length"] = df["tweet_length"] + df["log_length"] + 10
    prompt_length = config.prompt_length
    if config.model_name != "text-davinci-002" and prompt_length > 2000:
        prompt_length = 2000
    fewshot_df = make_few_shot_dataset(
        df, prompt_length=prompt_length, shuffled=config.shuffle_prompts
    )
    fewshot_df["prompt_length"] = get_tokenized_len(fewshot_df["prompt"].tolist())
    return pd.concat([df, fewshot_df], join="inner", axis=1)


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
    df = cleaned_df.copy()
    df = load_experiment_dataset(df)
    df["predictions"] = df["prompt"].map(run_inference)
    df[["bleu", "rogue", "bert_f1", "mean_score"]] = df.apply(
        lambda x: evaluate_inference(
            predictions=x["predictions"], references=x["cleaned_tweet"]
        ),
        axis=1,
    )
    wandb.log({"mean_score": df["mean_score"].mean()})
    predictions_table = wandb.Table(dataframe=df)
    predictions_artifact = wandb.Artifact("predictions", type="dataset")
    predictions_artifact.add(predictions_table, f"fewshot_predictions-{wandb.run.name}")
    wandb.log_artifact(predictions_artifact)
    wandb.log({"fewshot_predictions": predictions_table})
    return df


if __name__ == "__main__":
    main()
