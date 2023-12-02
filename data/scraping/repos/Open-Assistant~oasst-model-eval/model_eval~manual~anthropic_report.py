import json
import argparse
import datetime

import tqdm
import anthropic


def parse_args():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Enter message into URL')
    parser.add_argument('--api_key', type=str, default=None, help='URL to open', required=True)
    parser.add_argument('--input_file', type=str, default="data/prompt_lottery_en_250_text.jsonl", help='Path to input file')
    parser.add_argument('--model_name', type=str, default="claude-v1.3", help='Model name ')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--verbose', type=bool, default=True, help='Print output')

    # Parse arguments
    args = parser.parse_args()
    return args

def read_input(input_file):
    with open(input_file, "r") as f:
        lines = f.readlines()
        objs = [json.loads(line) for line in lines]
    return objs

def get_response(client, message, **params):
    output = client.completion(
        prompt=f"{anthropic.HUMAN_PROMPT} {message}{anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        **params
    )
    return output["completion"]


def main(args):
    # Read input file
    prompts = read_input(args.input_file)
    client = anthropic.Client(args.api_key)

    args_dict = dict(**args.__dict__)
    del args_dict["api_key"]
    del args_dict["verbose"]

    results = []
    sampling_report = {
        "model_name": args.model_name,
        "date": datetime.datetime.now().isoformat(),
        "args": args_dict,
        "prompts": results,
    }

    sampling_config = "sample"
    sampling_params = {
        "temperature": 0.8,
        "top_p": 1.0,
        "model": args.model_name,
        "max_tokens_to_sample": 512,
        "frequency_penalty": 0.0,
    }

    output_file = f"{datetime.datetime.now().strftime('%Y-%m-%d')}_{sampling_report['model_name']}_{sampling_config}.jsonl"

    try:
        for prompt in tqdm.tqdm(prompts):
            outputs = []
            for i in range(args.num_samples):
                response = get_response(client, prompt, **sampling_params)
                outputs.append(response)

                if args.verbose:
                    print(f"===[ Config: {sampling_config} [{i+1}/{args.num_samples}] ]===\n")
                    print(f'User: "{prompt}"')
                    print(f'Assistant: "{response}"\n')

            results.append({
                "prompt": prompt,
                "results": [{
                    "sampling_config": sampling_config,
                    "sampling_params": sampling_params,
                    "outputs": [response],
                }]
            })
    finally:
        with open(output_file, "w") as f:
            json.dump(sampling_report, f, indent=2)


if __name__ == '__main__':
    args = parse_args()
    main(args)
