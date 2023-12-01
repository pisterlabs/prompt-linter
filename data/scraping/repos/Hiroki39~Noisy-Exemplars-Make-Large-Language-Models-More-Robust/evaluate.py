import argparse
from datasets import load_dataset
from utils import model_evaluate, print_model_inputs
from uuid import uuid4
from dotenv import load_dotenv
import os
import csv


def conduct_test(
    model_name, dataset_name, prompt, shots, perturb, perturb_exemplar, dev
):
    run_id = str(uuid4())

    print(f"Run ID: {run_id}")

    # Load the GSM8K dataset from Hugging Face
    if dataset_name == "gsm8k":
        dataset = load_dataset(
            dataset_name,
            "main" if prompt != "ltm" else "socratic",
            download_mode="force_redownload",
        )
    elif dataset_name == "strategyqa":
        dataset = load_dataset(
            "ChilleD/StrategyQA",
            download_mode="force_redownload",
        )

    # Set up the OpenAI API client
    if model_name in ("gptturbo", "gpt3"):
        import openai

        openai.api_key = os.getenv("OPENAI_API_KEY")

    if model_name in ("gptturbo", "gpt3", "llama"):
        model_evaluate(
            run_id,
            model_name,
            dataset,
            dataset_name,
            prompt,
            shots,
            perturb,
            perturb_exemplar,
            dev,
        )
    elif model_name == "none":
        output_filename = (
            "generated_prompts/{}_{}_{}shots_perturb{}_{}perturbexemplar.json".format(
                dataset_name,
                prompt,
                shots,
                "none" if perturb == None else perturb,
                perturb_exemplar,
            )
        )
        print_model_inputs(
            run_id,
            model_name,
            dataset,
            dataset_name,
            prompt,
            shots,
            perturb,
            perturb_exemplar,
            dev,
            output_filename,
        )

    if not dev:
        with open("log_files.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    run_id,
                    model_name,
                    dataset_name,
                    prompt,
                    shots,
                    perturb,
                    perturb_exemplar,
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, default="gptturbo")
    parser.add_argument("--dataset", type=str, required=True, default="gsm8k")
    parser.add_argument("--prompt", type=str, required=True, default="cot")
    parser.add_argument("--shots", type=int, required=True, choices=[1, 2, 4, 8])
    parser.add_argument("--perturb", type=str, required=False)
    parser.add_argument(
        "--perturb_exemplar", type=int, required=True, choices=[0, 1, 2, 4, 8]
    )
    parser.add_argument("--dev", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.perturb_exemplar > args.shots:
        raise ValueError(
            "Number of perturbed exemplars cannot be greater than the number of shots"
        )

    if not args.perturb:
        args.perturb_exemplar = 0

    print("Current Arguments: ", args)

    load_dotenv()

    conduct_test(
        args.model,
        args.dataset,
        args.prompt,
        args.shots,
        args.perturb,
        args.perturb_exemplar,
        args.dev,
    )
