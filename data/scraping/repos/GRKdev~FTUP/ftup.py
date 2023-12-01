import os
import sys
from pyfiglet import Figlet
import re
import time
import argparse
import openai
import json
from collections import defaultdict
from dotenv import load_dotenv
import tiktoken
from clint.textui import colored
import keyboard
import threading

encoding = tiktoken.get_encoding("cl100k_base")

load_dotenv(".env")
figlet = Figlet()

parser = argparse.ArgumentParser(
    description="FT-UP uploading script for OpenAI FineTuning. Babbage and GPT-3.5"
)
# Using argparse to do command lines, Documentation https://docs.python.org/3/library/argparse.html

parser.add_argument(
    "-k",
    "--key",
    help="Usage: -k <key> or --key <key>. This argument is used to pass the API key. For env: default,must have env file with OPENAI_API_KEY=your_api_key_here",
    default="env",
    type=str,
)
parser.add_argument(
    "-m",
    "--model",
    help="Usage: -m <model> or --model <model>. Specifies the model to use ('gpt' for gpt-3.5-turbo-0613 or 'bab' for babbage-002)",
    type=str,
    required=True,
)
parser.add_argument(
    "-f",
    "--file",
    help="Usage: -f <file.csv> or --file <file.jsonl>. This argument is used to define what file use. In same folder.",
    type=str,
    required=True,
)
parser.add_argument(
    "-s",
    "--suffix",
    default="",
    help="Usage: -s <suffix> or --suffix <suffix>. This argument is used to add a Suffix for your finetuned model. (Example: 'my suffix title v-1')",
    type=str,
)
parser.add_argument(
    "-e",
    "--epoch",
    default=3,
    help="Usage: -e <epoch> or --epoch <epoch>. This argument is used to change the epoch for training. (1-50)",
    type=int,
)


def main():
    # Puting args inside main so i could make pytest into functions
    args = parser.parse_args()

    # Setting Tittle with Figlet
    figlet.setFont(font="slant")
    print(colored.red(figlet.renderText("FT - UP")))

    try:
        # Checking the OpenAI API Key
        if "env" in args.key:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            print(check_key(openai.api_key))

        else:
            print(check_key(args.key))
            openai.api_key = args.key

    except ValueError as e:
        sys.exit(e)

    except TypeError:
        sys.exit("Please check if OPENAI_API_KEY is set in .env file ⚠️")

    try:
        # Checking the Model name form arguments
        print(check_model(args.model))

    except ValueError as e:
        sys.exit(e)

    try:
        # Checking the CSV, if exists and have correct Headers
        print(check_jsonl_file(args.file))

    except (ValueError, FileNotFoundError) as e:
        sys.exit(e)

    # Creating the JSONL file and storing the file id name
    file_id_name = create_update_jsonl_file(args.model, args.file, args.epoch)

    # Creating the Finetuning Job.
    update_ft_job(file_id_name, args.model, args.suffix, args.epoch)


def check_key(key):
    # Checking key if not 51 lenght or starts with sk- and charcaters exit the program, if good -> set api key to openai module
    print(f"Checking API key format ...")
    match = re.search(r"^sk-[a-zA-Z0-9]+$", key)
    if match is None or len(key) != 51:
        raise ValueError(
            "\nInvalid format API key, You can find your API key at https://platform.openai.com/account/api-keys ⚠️"
        )
    else:
        return colored.green("- API Key")


def check_model(model):
    # checking the model if not exit the program
    print(f"\nChecking model ...")

    if not model in ["bab", "gpt"]:
        raise ValueError("\nInvalid model name ⚠️\nUsage: 'gpt' or 'bab'")
    else:
        return colored.green(f"- Model {model}")


def check_jsonl_file(file):
    # checking with regex if the name its correct.
    print(f"\nChecking if jsonl is valid ...")
    csv_file = re.search(r"^\w+\.jsonl$", file)

    if csv_file is None:
        # if we don't have match file must be wrong, need to use ValueError and not sys.exit
        raise ValueError(f"\nInvalid JSONL File: {file} ⚠️")

    if not os.path.exists(file):
        # Need it to import os to check if file exists wen we had match
        raise FileNotFoundError(f"\nFile {file} does not exist ⚠️")

    return colored.green(f"- JSON File {file}")


def create_update_jsonl_file(model, file, epoch):
    # Use the correct function to change csv into jsonl depending on model selected.
    if model == "gpt":
        jsonl_file_path = check_jsonl_gpt35(file)
    else:
        jsonl_file_path = check_jsonl_babbage(file)

    print("\nUploading jsonl train file ...")
    try:
        # Using OpenAI API to create a training File and storing ID Value.
        response = openai.File.create(
            file=open(jsonl_file_path, "rb"), purpose="fine-tune"
        )
        file_id_name = response["id"]
        print(f"- File ID: {colored.green(file_id_name)}")

        #  if model is gpt, we check the cost of the training
        if model == "gpt":
            cost_gpt(file, epoch)

    # Exceptions we could have, from OpenAI Documentation: https://platform.openai.com/docs/guides/error-codes/python-library-error-types
    except openai.error.APIError as e:
        # Handle API error here, e.g. retry or log
        sys.exit(e)

    except openai.error.APIConnectionError as e:
        # Handle connection error here
        sys.exit(e)

    except openai.error.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        sys.exit(e)

    except openai.error.AuthenticationError as e:
        # Handle Authentication Error
        sys.exit(f"\n{e} ⚠️")
    except openai.error.InvalidRequestError as e:
        # Handle invalid request format file
        sys.exit(e)

    return file_id_name


def check_for_cancel(job_id):
    keyboard.wait("x")
    try:
        openai.FineTuningJob.cancel(job_id)
        print("\nFine-tuning job was cancelled by user. ❌")

    except Exception as e:
        print(f"\nError cancelling the fine-tuning: {e}")


def update_ft_job(file_id_name, model, suffix, epoch):
    # Calling nested function update json and storing the file id name

    print("\nCreating a finetuning job ...")
    # Changing the model name to a correct model for creating de Fine Tunning Job.
    if model == "gpt":
        model = "gpt-3.5-turbo-0613"
    else:
        model = "babbage-002"

    response = openai.FineTuningJob.create(
        # Calling OpenAPI key with file id name from update_jsonl_files and the arguments from command line. Documentation: https://platform.openai.com/docs/api-reference/fine-tuning/create
        training_file=file_id_name,
        model=model,
        hyperparameters={
            "n_epochs": epoch,
        },
        suffix=suffix,
    )

    id = response["id"]
    # Storing Fine Tuning Job Name

    print(f"- Fintetuning job id: {colored.green(id)}\n")

    ft_model_name = None

    cancel_thread = threading.Thread(target=check_for_cancel, args=(id,))
    cancel_thread.start()

    while True:
        # Creating a loop for check the status of the training job until ends, cancelled or failed.
        try:
            # Calling in loop to retriever API and getting the status and Ft final model name if exists. Documentation: https://platform.openai.com/docs/api-reference/fine-tuning/retrieve
            response_job = openai.FineTuningJob.retrieve(id)
            status = response_job["status"]
            ft_model_name = response_job["fine_tuned_model"]
            print(f"\r{' '*50}", end="")
            # Asked the Duck: Printing 50 charcaters to erase the line
            print(f"\rStatus: {status} (Press 'x' to cancel)", end="")
            # With \r we can Overwrite the line, preventing new lines.

            if status in ["succeeded"]:
                # Breaking the loop and printing result
                sys.exit(
                    f"\nFinetuning {colored.green(status)}! ☑️\nFinetune model: {ft_model_name}\n"
                )
            elif status in ["failed", "cancelled"]:
                # Breaking the loop and printing result
                sys.exit(f"\nFinetuning {colored.red(status)}! ❌\n")

        # Code snippets from Documentation OpenAI
        except openai.error.APIError as e:
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.error.APIConnectionError as e:
            # Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            pass
        except openai.error.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass

        time.sleep(20)


def check_jsonl_gpt35(file):
    print(f"\nChecking if jsonl format is valid for GPT-3.5 training ...")

    data_path = file
    dataset = []

    # Load the dataset
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):  # Start line numbering from 1
            try:
                parsed_line = json.loads(line)
                dataset.append(parsed_line)
                if "messages" not in parsed_line:
                    raise KeyError("'messages' key missing")

            except (KeyError, json.JSONDecodeError) as e:
                sys.exit(f"\nError decoding JSON on line {line_num}: {e} ❌")

    # Checking format https://cookbook.openai.com/examples/chat_finetuning_data_prep
    print("- Num examples:", colored.yellow(len(dataset)))

    format_errors = defaultdict(int)

    # Check each example in the dataset
    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        # Check each message in the example (list) for correct format and content.
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            if not content or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    # Print out any errors found
    if format_errors:
        print(colored.red("\nFound errors:"))
        for k, v in format_errors.items():
            print(f"{k}: {v}")
        sys.exit("Not valid data format for GPT-3.5 training. Exiting... ❌\n")

    else:
        print(colored.green(f"- JSONL {file} correct format"))
        return data_path


def check_jsonl_babbage(file):
    print(f"\nChecking if jsonl format is valid for Babbage-002 training ...")
    data_path = file
    dataset = []
    # Load the dataset
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):  # Start line numbering from 1
            try:
                dataset.append(json.loads(line))
            except (KeyError, json.JSONDecodeError) as e:
                sys.exit(
                    f"\nError decoding JSON on line {colored.red(line_num)}: {e} ❌"
                )

    # Initial dataset stats
    print("- Num examples:", colored.yellow(len(dataset)))

    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        prompt = ex.get("prompt", None)
        completion = ex.get("completion", None)

        if not prompt:
            format_errors["missing_prompt"] += 1

        if not completion:
            format_errors["missing_completion"] += 1

        if not isinstance(prompt, str):
            format_errors["invalid_prompt_type"] += 1

        if not isinstance(completion, str):
            format_errors["invalid_completion_type"] += 1

    if format_errors:
        print(colored.red("\nFound errors:"))
        for k, v in format_errors.items():
            print(f"{k}: {v}")
        sys.exit("Not valid data format for Babbage-002. Exiting... ❌")
    else:
        print(colored.green(f"- JSONL {file} correct format"))
        return data_path


# Costumed version of https://cookbook.openai.com/examples/chat_finetuning_data_prep
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def cost_gpt(file, epochs):
    with open(file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    MAX_TOKENS_PER_EXAMPLE = 4096

    n_billing_tokens_in_dataset = sum(
        min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens
    )
    cost_per_token = 0.0080 / 1000  # Convert $0.0080 per 1K tokens to cost per token
    total_tokens = epochs * n_billing_tokens_in_dataset
    total_cost = total_tokens * cost_per_token

    print(
        f"\nDataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training"
    )
    print(f"You'll train for {colored.yellow(epochs)} epochs on this dataset")
    print(f"By default, you'll be charged for ~{colored.yellow(total_tokens)} tokens")
    print(f"Total cost: {colored.yellow(f'${total_cost:.4f}')}" + " 💰")


if __name__ == "__main__":
    main()
