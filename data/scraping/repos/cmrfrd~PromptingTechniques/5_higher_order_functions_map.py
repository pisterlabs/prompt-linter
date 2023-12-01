import asyncio
import enum
from typing import TYPE_CHECKING, Optional

import anyio
import instructor
import numpy as np
import openai
import pandas as pd
import tiktoken
import typer
from instructor.patch import wrap_chatcompletion
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from tenacity import retry, wait_random_exponential
from tqdm import asyncio as tqdm_asyncio

from prompting_techniques import AsyncTyper, execute, format_prompt

np.random.seed(1)

client = openai.AsyncOpenAI()
func = wrap_chatcompletion(client.chat.completions.create)
app = AsyncTyper()

## Pre process dataset
## https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset
percent_of_data = 0.005
newsgroups = fetch_20newsgroups(
    subset="all", data_home="./data", remove=("headers", "footers", "quotes")
)
newsgroups_target_names = [newsgroups.target_names[i] for i in newsgroups.target]  # type: ignore

percent_of_data = 0.1
num_targets = 5

target_names_small = np.random.choice(newsgroups.target_names, num_targets, replace=False)  # type: ignore
df = pd.DataFrame({"text": newsgroups.data, "target": newsgroups_target_names})  # type: ignore
df["text_short"] = df["text"].apply(lambda x: x[:5000])
test_df = df[df["target"].isin(target_names_small)].sample(frac=percent_of_data, random_state=1)

NewsTopic = enum.Enum("NewsTopic", [(t, t) for t in target_names_small])


class InferedNewsTopic(BaseModel):
    topic: "NewsTopic" if TYPE_CHECKING else NewsTopic  # type: ignore


semaphore = asyncio.Semaphore(128)


@retry(wait=wait_random_exponential(multiplier=1, max=3))
async def classify_article(article: str) -> InferedNewsTopic:
    async with semaphore:
        result: Optional[InferedNewsTopic] = None
        for attempt in range(3):
            try:
                result = await asyncio.wait_for(
                    func(
                        messages=[
                            {
                                "role": "user",
                                "content": format_prompt(
                                    f"""
                            You are an AI data labeler. You have one goal: to classify a given news article into one of the following topics: {', '.join(target_names_small)}.
                            
                            The topics are abbreviated. Remember to think how the full article relates to the topic.
                            
                            Think deeply and step by step about which topic this article belongs to. Check your work carefully.
                            
                            Here is the article:
                            {article}
                            
                            The topic of this article is one of the following "{', '.join(target_names_small)}". What is the topic of this article? Please output just the topic and nothing else.
                            """
                                ),
                            },
                        ],
                        model="gpt-3.5-turbo-0613",
                        response_model=InferedNewsTopic,
                        temperature=0.1,
                        seed=256,
                    ),
                    timeout=1,
                )
                break
            except asyncio.TimeoutError:
                continue
        if result is None:
            raise RuntimeError("Failed to classify article after 3 attempts")
    return InferedNewsTopic.model_validate(result)


@app.command()
async def map_example():
    """Run an example support request label task."""
    typer.echo("Running map' based classification example on newsgroups dataset.")
    typer.echo("\n")
    typer.echo("Dataset stats:")
    typer.echo(f"  - Number of articles: {len(test_df)}")
    typer.echo(f"  - Topics: {', '.join(target_names_small)}")
    typer.echo("\n")

    coros = [classify_article(article) for article in test_df["text_short"].tolist()]
    tasks = [asyncio.create_task(coro) for coro in coros]
    results = list(await execute(tasks, desc="Classifying articles"))

    test_df["inferred_target"] = list(map(lambda i: i.topic.value, results))
    typer.echo("\n\n\n")

    num_correct = (test_df["target"] == test_df["inferred_target"]).sum()
    typer.echo(f"Number correct: {num_correct}")
    typer.echo(f"Number incorrect: {len(test_df) - num_correct}")
    accuracy = (test_df["target"] == test_df["inferred_target"]).mean()
    typer.echo(f"Accuracy: {accuracy}")
    precision = precision_score(test_df["target"], test_df["inferred_target"], average="macro")
    typer.echo(f"Precision: {precision}")
    recall = recall_score(test_df["target"], test_df["inferred_target"], average="macro")
    typer.echo(f"Recall: {recall}")

    typer.echo("\n\n")
    typer.echo("Confusion matrix:")
    cm = confusion_matrix(test_df["target"], test_df["inferred_target"])
    labels = sorted(list(set(test_df["target"]) | set(test_df["inferred_target"])))
    typer.echo(pd.DataFrame(cm, index=labels, columns=labels).to_markdown())


if __name__ == "__main__":
    app()
