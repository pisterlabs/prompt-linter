from pathlib import Path
import time
import os

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tiktoken
import openai
import srsly
import typer

MODELS_COST = {
    "text-davinci-003": 0.0200,
    "text-curie-001": 0.0020,
    "text-babbage-001": 0.0005,
    "text-ada-001": 0.0004,
}

app = typer.Typer()


def gpt(model, text, labels):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = f"Classify the text into one of the following labels: {', '.join(labels)}\n\nText: {text}\nLabel:"

    retries = 0
    while retries < 3:
        try:
            response = openai.Completion.create(
                model=model, prompt=prompt, temperature=0, max_tokens=7
            )
        except:
            print("OpenAI error.")
            sleep_time = 2**retries
            print(f"Waiting for {sleep_time} seconds")
            time.sleep(sleep_time)
            print("Retrying...")
        else:
            break
        retries += 1
    return response["choices"][0]["text"].strip()


@app.command()
def predict(
    data_path, pred_data_path, model="text-davinci-003", sample_size: int = 100
):
    data = list(srsly.read_jsonl(data_path))

    if sample_size:
        data = data[:sample_size]

    labels = list(set([example["label"] for example in data]))

    encoding = tiktoken.encoding_for_model(model)
    num_tokens = sum([len(encoding.encode(example["text"])) for example in data])

    price = num_tokens * MODELS_COST[model] / 1000
    if typer.confirm(f"This will cost you {price:.2f}$. Do you want to continue?"):
        pred_data = []
        for example in tqdm(data):
            pred_label = gpt(model, example["text"], labels)
            pred_data.append({"text": example["text"], "label": pred_label})
            # OpenAI rate limit is 1 per second
            time.sleep(1)

        srsly.write_jsonl(pred_data_path, pred_data)


@app.command()
def evaluate(data_path, pred_data_path, result_path: Path, sample_size: int = 100):
    data = list(srsly.read_jsonl(data_path))

    if sample_size:
        data = data[:sample_size]

    pred_data = list(srsly.read_jsonl(pred_data_path))

    y_true = [example["label"] for example in data]
    y_pred = [example["label"] for example in pred_data]
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    srsly.write_json(result_path, {"accuracy": accuracy})


if __name__ == "__main__":
    app()
