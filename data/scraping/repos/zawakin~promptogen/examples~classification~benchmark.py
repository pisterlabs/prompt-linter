import time
from typing import List

import typer

import promptogen as pg

# from tenacity import retry, stop_after_attempt
from examples.base import make_output_path
from examples.classification.dataset_loader import DatasetLoader, IMDbSentimentDataset, TweetEvalEmotionDataset
from examples.llm.openai_util import OpenAITextLLM

app = typer.Typer(add_completion=True)


@app.command("tweet_eval_emotion")
def run_tweet_eval_emotion():
    dataset = TweetEvalEmotionDataset(seed=43)
    benchmark_by_dataset(dataset)


@app.command("imdb_sentiment")
def run_imdb_sentiment():
    dataset = IMDbSentimentDataset(seed=43)
    benchmark_by_dataset(dataset)


formatter = pg.KeyValuePromptFormatter()
llm = OpenAITextLLM(model="gpt-3.5-turbo-16k")
prompt_runner = pg.TextLLMPromptRunner(llm=llm, formatter=formatter)


def run_benchmark_prompt(prompt_to_test: pg.Prompt, test_examples: List[pg.IOExample], output_key: str):
    total = 0
    correct = 0

    for example in test_examples:
        time.sleep(3)

        # @retry(stop=stop_after_attempt(2))
        def f() -> pg.Value:
            return prompt_runner.run_prompt(prompt_to_test, example.input)

        try:
            resp = f()
        except Exception as e:
            print(f"Error: {e}; skipping...")
            total += 1
            continue

        total += 1

        got = resp[output_key]
        expected = example.output[output_key]

        if got == expected:
            correct += 1
        else:
            print(f"got {got} expected {expected}, input: {example.input}")

        print(f"Total: {total}, Correct: {correct}, Accuracy: {correct/total}")


def benchmark_by_dataset(dataset: DatasetLoader):
    # dataset = TweetEvalEmotionDataset(seed=43)
    dataset = IMDbSentimentDataset(seed=43)

    prompt_to_test = pg.Prompt.from_json_file(make_output_path(dataset.attributes.name + ".json"))
    # prompt_to_test = pg.Prompt.from_json_file(make_json_path(dataset.attributes.name + '_with_reason.json'))

    test_examples = dataset.load_test_dataset()

    run_benchmark_prompt(prompt_to_test, test_examples, dataset.attributes.output_key)


if __name__ == "__main__":
    app()
