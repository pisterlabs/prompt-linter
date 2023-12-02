from joblib import Memory
from collections import defaultdict
import openai
import time
import os
import glob
from typing import Tuple, List
from toearthly.core import constants

memory = Memory(location='data/gpt_cache', verbose=1)

openai.api_key = os.getenv('OPENAI_API_KEY')

@memory.cache
def call_chat_completion_api_cached(max_tokens, messages,temperature):
    print("running prompt")
    return call_chat_completion_api(max_tokens,messages, temperature)


def call_chat_completion_api(max_tokens, messages,temperature):
    max_retries = 3
    initial_delay = 1
    factor = 2

    retries = 0
    delay = initial_delay

    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )
            print(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}, Retrying...")
            time.sleep(delay)
            retries += 1
            delay *= factor
    print("Max retries reached. Returning 'Error'.")
    return "Error: Max Retry Hit"

def read(filepath: str) -> str:
    with open(filepath, 'r') as outfile:
        return outfile.read()

def relative_read(relative_filepath: str) -> str:
    # Get the directory of the current script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full filepath, by going up one folder
    full_filepath = os.path.join(script_dir, ".." ,relative_filepath)

    with open(full_filepath, 'r') as outfile:
        return outfile.read()

def write(contents: str, filepath: str) -> None:
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, 'w') as outfile:
        outfile.write(contents)


def write_debug(filename: str, contents: str) -> None:
    filepath = os.path.join(constants.DEBUG_DIR, filename)
    os.makedirs(constants.DEBUG_DIR, exist_ok=True)
    with open(filepath, 'w') as outfile:
        outfile.write(contents)

def find_first_yml(path=None) -> Tuple[str,str]:
    if path is None:
        path = os.getcwd()
        
    if not path.endswith("/"):
        path += "/"

    yml_files = glob.glob(path + ".github/workflows/*.yml")

    if not yml_files:
        raise Exception("No yml files found. Process will stop.")

    with open(yml_files[0], 'r') as file:
        yml = file.read()
    write_debug("workflow.yml", yml)
    return (yml,yml_files[0])

def find_workflows(path=None) -> List[str]:
    if path is None:
        path = os.getcwd()
        
    if not path.endswith("/"):
        path += "/"

    yml_files = glob.glob(path + ".github/workflows/*.yml")

    if not yml_files:
        raise Exception("No yml files found. Process will stop.")

    return yml_files
    

def find_first_dockerfile(path=None) -> str:
    if path is None:
        path = os.getcwd()
        
    if not path.endswith("/"):
        path += "/"

    docker_files = glob.glob(path + "Dockerfile")

    if not docker_files:
        return ""

    with open(docker_files[0], 'r') as file:
        dockerfile = file.read()
    write_debug("Dockerfile", dockerfile)
    return dockerfile


# Like tree but less output
def print_directory(path, prefix='', level=0, max_level=1) -> str:
    if level > max_level:
        return ''

    dir_structure = ''
    dir_items = defaultdict(list)

    # Group files by extension and directories separately
    for item in os.listdir(path):
        # Ignore hidden files and directories
        if item.startswith('.'):
            continue

        if os.path.isfile(os.path.join(path, item)):
            ext = os.path.splitext(item)[1]
            dir_items[ext].append(item)
        else:
            dir_items['folders'].append(item)

    # Generate directory structure, combining files with same extension if more than 3
    for ext, items in dir_items.items():
        if ext != 'folders':
            if len(items) > 3:
                dir_structure += f"{prefix}├── *{ext}\n"
            else:
                for item in items:
                    dir_structure += f"{prefix}├── {item}\n"
        else:
            for item in items:
                dir_structure += f"{prefix}├── {item}/\n"
                if level < max_level:
                    subdir_structure = print_directory(os.path.join(path, item),
                                                       prefix + "│   ",
                                                       level + 1, 
                                                       max_level)
                    dir_structure += subdir_structure
    write_debug("files.txt", dir_structure)
    return dir_structure

def log(message: str) -> None:
    log_file_path = os.path.join(constants.DEBUG_DIR, 'log.txt')
    os.makedirs(constants.DEBUG_DIR, exist_ok=True)

    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')

