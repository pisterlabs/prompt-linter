import openai
import os
import argparse
from getpass import getpass

def main(args):
    openai.api_key = args.api_key

    os.environ['OPENAI_API_KEY'] = args.api_key

    prepare_data_command = f"openai tools fine_tunes.prepare_data -f {args.filename}"
    os.system(prepare_data_command)

    # Prepare OpenAI jsonl
    base_filename = os.path.splitext(args.filename)[0]
    train_data = f"{base_filename}_prepared.jsonl"

    # Call finetune
    create_command = f"openai api fine_tunes.create -t {train_data} -m {args.model}"
    os.system(create_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI API script")
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--filename", type=str, help="File name for prepare_data command")
    parser.add_argument("--model", type=str, help="Model for fine_tunes.create command")
    args = parser.parse_args()

    main(args)
