import sys
import yaml
import requests
import json
import os
import itertools

import openai_wrapper

from itertools import product

def generate_token_combinations(token_files):
    """
    This function generates all possible combinations of tokens from a list of files.

    Args:
        token_files: A list of files containing tokens.

    Returns:
        A list of lists of tokens, where each list represents a possible combination.
    """

    # Create a list to store all tokens from all files
    all_tokens = []

    # Iterate over all files
    for file in token_files:
        # Open the file and read the tokens
        with open(file, "r") as f:
            tokens = f.read().splitlines()
        # Add the list of tokens from the current file to the main list
        all_tokens.append(tokens)

    # Generate all possible combinations of tokens and return
    return list(product(*all_tokens))


def generate_token_permutations1(token_files):
    """
    This function generates all possible permutations of tokens from a list of files.

    Args:
        token_files: A list of files containing tokens.

    Returns:
        A list of lists of tokens, where each list represents a possible permutation.
    """

    # Check if there is only one file. If so, return a list of all the tokens in the file.
    if len(token_files) == 1:
        # Open the file and read the tokens.
        with open(token_files[0], "r") as f:
            tokens = f.read().splitlines()

        # Return a list containing a single list of tokens.
        return [tokens]

    # Otherwise, get the first file and the remaining files.
    current_file = token_files[0]
    remaining_files = token_files[1:]

    # Get the tokens from the current file.
    with open(current_file, "r") as f:
        current_tokens = f.read().splitlines()

    # Generate all possible permutations of the tokens from the remaining files.
    remaining_permutations = generate_token_permutations(remaining_files)

    # Create a list of all possible permutations by adding each token from the current file to each permutation from the remaining files.
    permutations = [
        [current_token] + permutation
        for current_token in current_tokens
        for permutation in remaining_permutations
    ]

    # Return the list of permutations.
    return permutations


def process_permutation(template, tokens):
    result = template
    for i, token in enumerate(tokens, start=1):
        placeholder = "{" + str(i) + "}"
        result = result.replace(placeholder, token)
    return result

def create_filename(prefix="",suffix="", elements=[], extension=".txt"):
    filename = prefix + "_".join(elements) + suffix + extension
    return filename


def main():
    # Read command-line arguments
    config_file = sys.argv[1]

    # Parse the YAML config file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract values from the config
    system_file = config["system_file"]
    user_file = config["user_file"]
    token_files = config["token_files"]
    gpt_temp = config["gpt_temp"]
    gpt_model= config["gpt_model"]
    gpt_functions_file = config["gpt_functions_file"]


    # Read the template file
    with open(user_file, "r") as f:
        template = f.read()

    # Read tokens from token files
    tokens_list = generate_token_combinations(token_files)
    print(tokens_list)

    # Read the system role file
    with open(system_file, "r") as f:
        system_content = f.read()

    # Read the functions file
    with open(gpt_functions_file, "r") as f:
        openai_wrapper.functions = json.loads(f.read())

    # process_permutations
    for tokens in tokens_list:
        print(tokens)
        prompt=process_permutation(template, tokens)
        openai_wrapper.messages = []
        openai_wrapper.append_message_log("system", system_content)
        openai_wrapper.append_message_log("user", prompt)
        output=openai_wrapper.send_message(message_log=openai_wrapper.messages,model=gpt_model,temperature=gpt_temp,max_tokens=1000)
        filename=create_filename(prefix="output_",elements=tokens)
        with open(filename,'w') as f:
            f.write(output)

if __name__ == "__main__":
    main()
