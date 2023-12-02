#!/usr/bin/env python3

import argparse
import difflib
import glob
import json
import os
import openai
import re
import subprocess
import sys
import threading
import time
from pprint import pprint

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

global args
args = False

global g_debug
g_debug = False

_ = load_dotenv(find_dotenv()) # read local .env file

# sys.path.append('../..')

try:
    openai.api_key  = os.environ['OPENAI_API_KEY']
except KeyError:
    print("WARN: OPENAI_API_KEY environment variable not set.")

# gpt-3.5-turbo makes mistakes without additional docs
# llm_model = "gpt-3.5-turbo"

llm_model = "gpt-4"


def model_check():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    available_models = openai.Model.list()
    if not any(model["id"] == llm_model for model in available_models["data"]):
        error("Your API key doesn't have access to the gpt-4 model. See https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4")
    else:
        debug("gpt-4 model is available.")


def debug(msg):
    "Print a debug message if the global debug flag is set."
    if g_debug:
        print('DEBUG:', msg)


def error(msg):
    "Print an error message and exit."
    print('ERROR:', msg)
    sys.exit(1)


def parse_args():
    "Parse command line arguments."
    global args
    global g_debug

    description="""
Use gpt-4 to operate on all Terraform files in the current directory.\n\n
Expects OpenAI API key in OPENAI_API_KEY environment variable.
"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-op", "--operation", type=str, help="Operation to perform. Included in LLM prompt.", required=True)
    parser.add_argument("-d", dest="debug", action='store_true', help="Print debug information")
    parser.add_argument("--no-precheck", dest="no_precheck", action='store_true', default=False,
        help="Do not run 'terraform init' or 'terraform validate' before starting langchain processing.")
    parser.add_argument("--no-diff", dest="no_diff", action='store_true', default=False,
        help="Do not print diff between old and new files.")
    parser.add_argument("-dry", dest="dry", action='store_true', help="Dry run. Don't submit to OpenAI.")

    
    args = parser.parse_args()
    g_debug = args.debug
    operation = args.operation


def make_timestamp_tmp_dir():
    "Create a timestamped subdirectory. Return the full path."
    current_time = time.strftime('%Y%m%d_%H.%M.%S')
    prefix = 'tfai_output_'
    path = os.path.join(os.getcwd(), prefix + current_time)
    debug(f"Creating directory {path}")
    try:
        os.mkdir(path)
    except FileExistsError:
        print(f"ERROR: A directory with the name {current_time} already exists.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")

    return(path)    


def encode_tf_files_to_delim_str():
    "Encode all .tf files in the current directory into a single delimited string. Return string."
    tf_files = glob.glob('*.tf')
    output_string = ''
    
    for tf_file in tf_files:
        debug(f"Encoding {tf_file}")
        with open(tf_file, 'r') as file:
            content = file.read()
            output_string += f'---BEGIN {tf_file}---\n{content}\n---END {tf_file}---\n\n'
    
    return output_string


def recreate_tf_files_from_delim_str(delimited_str):
    "Recreate all .tf files in a temp directory from a single delimited string. Returns output dir path."

    tmpdir = make_timestamp_tmp_dir()

    # Split the string into segments based on the custom delimiter
    segments = re.split(r'---BEGIN (.+?)---|---END .+?---', delimited_str)
    # debug('Segments before removing empties')
    # if g_debug:
    #     for segment in segments:
    #         debug(f"ORIG SEGMENT: {segment}")
    #         debug('END ORIG SEGMENT')
    
    # Remove None and empty string elements from the list
    # segments = [segment for segment in segments if segment]
    # filtered_segments = [item for item in segments if item not in [None, '', '\n', '\n\n']]
    filtered_segments = [item for item in segments if item not in [None, '', '\n', '\n\n']]
    # debug('Segments after removing empties')
    # if g_debug:
    #     for segment in filtered_segments:
    #         debug(f"FILTERED SEGMENT: {segment}")
    #         debug('END FILTERED SEGMENT')

    
    # Create files from the segments
    for i in range(0, len(filtered_segments), 2):
        try:
            filename = filtered_segments[i].strip()
            # debug('Filename: ' + filename)
            content = filtered_segments[i + 1]
            # debug('content: ' + content)
        except IndexError:
            debug('IndexError, skipping')
            continue
        except AttributeError:
            debug('AttributeError, skipping')
            continue

        if not filename:
            debug("Skipping empty filename.")
            continue

        # Create the full path of the new file in the temporary directory
        full_path = os.path.join(tmpdir, filename)

        # Write the content to the new file
        debug(f"Writing {full_path}")
        with open(full_path, 'w') as f:
            f.write(content)
            f.write('\n')

        # debug(f"File {filename} recreated in {tmpdir}")
    return tmpdir


def get_completion_langchain():
    
    chat = ChatOpenAI(temperature=0.0, model=llm_model)

    # Curly braces in example TF code must be escaped by doubling them
    # so Langchain doesn't interpret them as placeholders.
    template_string = """
You are a Terraform developer working on a Terraform project with multiple HCL files ending in ".tf".
Your task is to perform the following operations on those files:
{operation} 

Do not make any other changes.
Each input filename is delimited with BEGIN and END lines like this:
---BEGIN filename.tf---
File contents
---END filename.tf---

Output all project files after your changes, using the same format as the input files.

Do not output anything other than the delimited files themselves.

Example file list format:
---BEGIN filename1.tf---
resource "random_pet" "pet1" {{
  length = 2
}}
---END filename1.tf---

---BEGIN filename2.tf---
resource "random_id" "server" {{
  byte_length = 8
}}
---END filename2.tf---

Input file list to process:
{tf_files_str}
"""
    
    prompt_template = ChatPromptTemplate.from_template(template_string)
    # debug(prompt_template.messages[0].prompt)

    tf_files_str = encode_tf_files_to_delim_str()

    messages = prompt_template.format_messages(operation=args.operation, tf_files_str=tf_files_str)

    debug('Submitting prompt to OpenAI API')
    response = chat(messages)
    content = response.content

    trimmed_resp = content[content.find("---BEGIN"):] if "---BEGIN" in content else content

    debug('BEGIN get_completion_langchain() trimmed_resp:')
    debug(trimmed_resp)
    debug('END get_completion_langchain() trimmed_resp')

    return trimmed_resp


def reader_thread(pipe, func):
    "Read lines from a pipe and call a function for each line. Part of a multi-threaded subprocess reader."
    while True:
        line = pipe.readline()
        if line:
            func(line.decode('utf-8').strip())
        else:
            break


def run_command(command):
    "Run a shell command and return the exit code."
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Define a function to handle output
    def print_output(line):
        if g_debug:
            print("Subprocess Output:", line)
    
    # Define a function to handle errors
    def print_error(line):
        print("Subprocess Error:", line)
    
    # Create threads for reading stdout and stderr
    out_thread = threading.Thread(target=reader_thread, args=[process.stdout, print_output])
    err_thread = threading.Thread(target=reader_thread, args=[process.stderr, print_error])
    
    # Start threads
    out_thread.start()
    err_thread.start()
    
    # Wait for both threads to finish
    out_thread.join()
    err_thread.join()
    
    # Wait for the process to finish and get the exit code
    process.communicate()
    return process.returncode


def terraform_init(after=False):
    "Run terraform init in the current directory. Return exit code."

    after_msg = ""
    if after:
        after_msg = "after langchain processing"
    debug("Running terraform init %s" % after_msg)
    command = "terraform init"
    exit_code = run_command(command)
    return exit_code


def terraform_validate(after=False):
    "Run terraform validate in the current directory. Return exit code."

    after_msg = ""
    if after:
        after_msg = "after langchain processing"
    debug("Running terraform validate %s" % after_msg)
    command = "terraform validate"
    exit_code = run_command(command) 
    return exit_code

# get_file_content() and compare_directories() are used for printing before/after diffs
def get_file_content(file_path):
    "Return the contents of a file."
    with open(file_path, 'r') as f:
        return f.readlines()

def compare_directories(dir1, dir2):
    "Compare .tf files between two directories and print differences."
    files1 = [f for f in os.listdir(dir1) if f.endswith('.tf')]
    files2 = [f for f in os.listdir(dir2) if f.endswith('.tf')]

    common_files = set(files1) & set(files2)
    only_in_dir1 = set(files1) - set(files2)
    only_in_dir2 = set(files2) - set(files1)

    if only_in_dir1:
        print(f"Files only in {dir1}: {', '.join(only_in_dir1)}")
    if only_in_dir2:
        print(f"Files only in {dir2}: {', '.join(only_in_dir2)}")

    for common_file in common_files:
        file1_content = get_file_content(os.path.join(dir1, common_file))
        file2_content = get_file_content(os.path.join(dir2, common_file))

        diff = difflib.unified_diff(file1_content, file2_content, fromfile=common_file, tofile=common_file)

        diff_output = list(diff)
        if diff_output:
            print(f"Differences in {common_file}:")
            print(''.join(diff_output))


def main():
    parse_args()

    model_check()

    if not args.no_precheck or args.dry:
        debug("Checking for terraform files in current directory")
        exit_code = terraform_init()
        # debug(f"terraform init exit code: {exit_code}")
        if exit_code != 0:
            error(f"terraform init failed with exit code {exit_code} before any operations. Please fix before running terrafai or use --skip-precheck.")
            sys.exit(1)

        exit_code = terraform_validate()
        # debug(f"terraform validate exit code: {exit_code}")
        if exit_code != 0:
            error(f"terraform validate failed with exit code {exit_code} before any operations. Please fix before running terrafai or use --skip-precheck.")
            sys.exit(1)

    new_tf_encoded = get_completion_langchain()
    output_dir = recreate_tf_files_from_delim_str(new_tf_encoded)

    os.chdir(output_dir)

    terraform_init(after=True)
    terraform_validate(after=True)

    os.chdir('..')

    # debug("if not args.no_diff...")
    if not args.no_diff:
        # debug("Comparing directories")
        compare_directories('.', output_dir)

    print("\nSUCCESS! New Terraform files generated in %s\n" % output_dir)


if __name__ == "__main__":
    main()

