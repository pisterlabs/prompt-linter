# Import necessary modules
import openai
import os
import sys
import time
import json
import re
import ast
from constants import DEFAULT_DIRECTORY, DEFAULT_MODEL, DEFAULT_MAX_TOKENS
from utils import clean_dir, write_file, get_file_content, get_file_paths, get_functions, chunk_and_summarize
from codingagents import design_agent, algorithm_agent, coding_agent, code_integration_agent, debug_agent, file_code_agent
from glob import glob
from openai.embeddings_utils import get_embedding
import pandas as pd
import numpy as np

from dotenv import load_dotenv
# Initialize OpenAI and GitHub API keys
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize a session with OpenAI's chat models
def chat_with_gpt3(systemprompt, prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": systemprompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9
    )
    return response['choices'][0]['message']['content']

def save_to_local_directory(repo_name, functions):
    # Check if the directory already exists
    if not os.path.exists(repo_name):
        # If not, create it
        os.makedirs(repo_name)
    
    # Create a new file in the directory to hold all the functions
    file_path = os.path.join(repo_name, "functions.py")
    with open(file_path, "w") as f:
        # Write all the functions to the file
        for function in functions:
            f.write(function)
            f.write("\n\n")

def generate_filepaths(prompt):
    systemprompt = f"""You are an AI developer who is trying to write a program that will generate code for the user based on their intent.

    When given their intent, create a complete, exhaustive list of filepaths that the user would write to make the program.
    
    Only list the filepaths you would write, and return them as a python array of strings. 
    do not add any other explanation, only return a python array of strings."""
    result = chat_with_gpt3(systemprompt, prompt)
    print(result)
    return result

def generate_filecode(filename, systems_design, filepaths_string, shared_dependencies=None, prompt=None):
    print("generating code")
    prompt = f"""
    We have broken up the program into per-file generation. 
    Now your job is to generate only the code for the file {filename}. 
    Make sure to have consistent filenames if you reference other files we are also generating.
    
    Remember that you must obey 3 things: 
       - you are generating code for the file {filename}
       - do not stray from the names of the files and the shared dependencies we have decided on
       - follow the {systems_design} laid out in the previous steps.
    
    Bad response:
    ```javascript 
    console.log("hello world")
    ```
    
    Good response:
    console.log("hello world")
    
    Begin generating the code now.

    """
    systemprompt = file_code_agent(filepaths_string, shared_dependencies)
    result = chat_with_gpt3(systemprompt, prompt)
    print(result)
    return filename, result


def generate_shared_dependencies(prompt, filepaths):
    systemprompt = f"""You are an AI developer who is trying to write a program that will generate code for the user based on their intent.
                
            In response to the user's prompt:

            ---
            the app is: {prompt}
            ---
            
            the files we have decided to generate are: {filepaths}

            Now that we have a list of files, we need to understand what dependencies they share.
            Please name and briefly describe what is shared between the files we are generating, including exported variables, data schemas, id names of every DOM elements that javascript functions will use, message names, and function names.
            Exclusively focus on the names of the shared dependencies, and do not add any other explanation.
    """
    result = chat_with_gpt3(systemprompt, prompt)
    print(result)
    return result

def debug_code(directory):
    extensions = ['py', 'html', 'js', 'css', 'c', 'rs']
    code_files = []

    for extension in extensions:
        code_files.extend(y for x in os.walk(directory) for y in glob(os.path.join(x[0], f'*.{extension}')))
    print("Total number of py files:", len(code_files))
    if len(code_files) == 0:
        print("Double check that you have downloaded the repo and set the code_dir variable correctly.")

    all_funcs = []
    for code_file in code_files:
        funcs = list(get_functions(code_file))
        for func in funcs:
            all_funcs.append(func)
    print("Total number of functions:", len(all_funcs))
    df = pd.DataFrame(all_funcs)
    df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, engine="text-embedding-ada-002")) 
    df['filepath'] = df['filepath'].apply(lambda x: x.replace(directory, ""))
    df.to_csv("functions.csv", index=True)
    df.head()
    debug_code_agent = chat_with_gpt3(debug_agent, all_funcs)


# Main function
def main(prompt, directory=DEFAULT_DIRECTORY, model=DEFAULT_MODEL, file=None):
    # Get the project objective from the user
    if prompt.endswith(".md"):
        with open(prompt, "r") as f:
            prompt = f.read()
    print("Hello, I am your local AI developer! You said you wanted:")
    print("\033[92m" + prompt + "\033[0m")
    # Get the repo name from the user
    repo_name = input("Enter the name for the new directory: ")
    directory = os.path.join(directory, repo_name)
    prompt_string = json.dumps(prompt)
    design_prompt = design_agent()
    systems_design = chat_with_gpt3(design_prompt, prompt_string)
    print(f"Systems design: "+systems_design)
    code_prompt = coding_agent()
    code = chat_with_gpt3(code_prompt, prompt+systems_design)
    print(f"code: "+code)
    code_integration_prompt = code_integration_agent()
    code_integration = chat_with_gpt3(code_integration_prompt, prompt+systems_design+code)
    filepaths = generate_filepaths(prompt)
    filepaths_string = json.dumps(filepaths)
    list_actual = []
    try:
        list_actual = ast.literal_eval(filepaths)
        shared_dependencies = None
        if os.path.exists("shared_dependencies.md"):
            with open("shared_dependencies.md", "r") as f:
                shared_dependencies = f.read()
        
        if file is not None:
            print("File", file)
            filename, filecode = generate_filecode(file, systems_design, filepaths_string, shared_dependencies)
            write_file(filename, filecode, directory)
        else:
            clean_dir(directory)

            shared_dependencies = generate_shared_dependencies(prompt, filepaths_string)
            write_file("shared_dependencies.md", shared_dependencies, directory)

            for filename, filecode in generate_filecode.map(
                list_actual, order_outputs=False, kwargs=dict(systems_design, filepaths_string, shared_dependencies)
            ):
                write_file(filename, filecode, directory)
        debug_code(directory)
    except ValueError:
        print("Failed to parse result")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        if not os.path.exists("prompt.md"):
            print("Please provide a prompt file or a prompt string")
            sys.exit(1)
        else:
            prompt = "prompt.md"

    else:
        prompt = sys.argv[1]

    directory = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DIRECTORY
    file = sys.argv[3] if len(sys.argv) > 3  else None
    main(prompt, directory, file)