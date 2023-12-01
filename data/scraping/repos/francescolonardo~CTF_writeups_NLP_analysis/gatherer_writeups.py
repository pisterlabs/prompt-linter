import argparse
import json
import string
import os
import time
import secrets

import openai

from settings import MODEL_ID, MODEL_RANDOMNESS_GATHERER
from settings import DATASET_DIRPATH, DATASET_TXT_SUFFIXES
from settings import NEW_WRITEUP_CMD, NONCE_LENGTH


def get_response(messages: list):
    response = openai.ChatCompletion.create(  # https://platform.openai.com/docs/api-reference/chat/create
        model=MODEL_ID,
        messages=messages,
        temperature=MODEL_RANDOMNESS_GATHERER,
        stream=False,
    )
    return response.choices[0].message


def generate_nonce(length=NONCE_LENGTH):
    # generate a random string of uppercase letters and digits
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def clean_writeup(file_content):
    # remove first line
    file_content = file_content.split("\n", 1)[1]
    # remove first line if it is empty
    lines = file_content.split("\n")
    if len(lines) > 1 and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def write_file(out_filepath, file_content):
    with open(out_filepath, "w", encoding="utf-8") as output_file:
        output_file.write(file_content)


if __name__ == "__main__":
    # create an argument for the integer parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest="number", type=int,
                        help="the number of writeups to gather")
    args = parser.parse_args()
    # get the number of writeups to scan
    num_writeups = args.number
    # check if the integer parameter is provided
    if not num_writeups:
        parser.error("an integer parameter is required")

    with open("./api_keys.json", "r", encoding="utf-8") as f:
        api_key = json.load(f)["api_key"]
    openai.api_key = api_key
    with open("./guidelines.json", "r", encoding="utf-8") as f:
        requirements = json.load(f)
        basic_requirements = requirements["basic_requirements"]
        gatherer_requirements = requirements["gatherer_requirements"]
    guidelines = basic_requirements + gatherer_requirements
    print("Sending the context...")
	#get_response(messages=guidelines) # TODO: remove this (?)
    
    try:
        print("Sending the writeups history...")
        # loop through all directories and subdirectories
        for dirpath, dirnames, filenames in os.walk(DATASET_DIRPATH):
            # loop through all files in the directory
            for filename_idx, filename in enumerate(filenames, start=1):
                # check if the file is a "*_original.txt" file
                if filename.endswith(DATASET_TXT_SUFFIXES[0]):
                    prefix_filename = filename.split(DATASET_TXT_SUFFIXES[0])[0]
                    if os.path.exists(os.path.join(DATASET_DIRPATH, prefix_filename+DATASET_TXT_SUFFIXES[2])):
                        with open(os.path.join(DATASET_DIRPATH, prefix_filename+DATASET_TXT_SUFFIXES[2]), "r", encoding="utf-8") as file:
                            # take the first five lines of a tagged writeup
                            file_summary = [line for i,
                                            line in enumerate(file) if i < 5]
                            file_summary = "".join(file_summary)
                        guidelines.append(
                            {"role": "user", "content": file_summary})
                        # five writeups unwanted at a time
                        if filename_idx % 5 == 0 or filename_idx == len(filenames):
                            get_response(messages=guidelines)

        print("Gathering new writeups...")
        for writeup_idx in range(num_writeups):
            guidelines.append({"role": "user", "content": NEW_WRITEUP_CMD})
            max_retries = 5
            retry_count = 0
            while retry_count < max_retries:
                try:
                    assistant_output = get_response(messages=guidelines)
                    guidelines.append(assistant_output)
                    # clean writeup content
                    file_content = clean_writeup(assistant_output["content"])
                    # generate a nonce
                    filename = f"writeup_{generate_nonce()}.txt"
                    if not os.path.exists(DATASET_DIRPATH):
                        os.makedirs(DATASET_DIRPATH)
                    out_filepath = os.path.join(DATASET_DIRPATH, filename)
                    write_file(out_filepath, file_content)
                    print(f"{writeup_idx+1}. Writeup file: {out_filepath}")
                    break  # exit the loop if the function call succeeds
                except openai.error.InvalidRequestError as e:
                    print(f"OpenAI API request is invalid: {e}")
                    retry_count += 1
                    if retry_count == max_retries:
                        break
                except openai.error.RateLimitError as e:
                    print(f"OpenAI API request exceeds rate limit: {e}")
                    time.sleep(60)  # sleep for 60 seconds
        print("Done.")
    except KeyboardInterrupt:
        print("Stop.")
