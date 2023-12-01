import json
import time
from argparse import ArgumentParser

import openai
import pandas as pd
from tqdm import tqdm

MODEL_TO_USE = "text-davinci-003"


def main(args):
    # Loading data
    data = pd.read_csv("dataset/llm_sampled_data.csv")
    # Setting API key
    with open("openai_key.txt", encoding="utf-8") as f:
        key = f.readline()
    openai.api_key = key
    # Loading prompt template
    with open("prompts_templates.json", encoding="utf-8") as f:
        prompt = json.load(f)["dietary_tags_classification"][
            args.prompt_number
        ]
    # Creating list of prompts to be sent to API
    model_inputs = [
        prompt.replace("INPUT_TEXT", text)
        for text in data.ingredients_list.iloc[:args.number_examples]
    ]

    responses = []
    # Getting responses for all inputs
    for text in tqdm(model_inputs):
        response = openai.Completion.create(
            model=MODEL_TO_USE,
            prompt=text,
            temperature=0,
            max_tokens=1024,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        responses += [x["text"] for x in response["choices"]]
        # Adding sleep because of API requests limits
        time.sleep(1)
    # Creating results dataframe and saving it
    results_dataframe = pd.DataFrame(
        {
            "ingredients_list": data.ingredients[: len(responses)].tolist(),
            "gpt-3_classification": responses,
        }
    )

    results_dataframe.to_csv(
        "results/unspecified_dietary_tags_classification_results.csv", index=False
    )


if __name__ == "__main__":
    # Adding argument parser.
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--number_examples",
        default=50,
        type=int,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "-p",
        "--prompt_number",
        default=0,
        type=int,
        help="Prompt number to use",
    )
    args = parser.parse_args()
    main(args)
