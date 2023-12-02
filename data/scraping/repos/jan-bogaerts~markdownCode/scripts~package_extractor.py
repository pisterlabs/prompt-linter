import sys
import os
from time import sleep
from constants import get_model_config, DEFAULT_MAX_TOKENS, OPENAI_API_KEY
import project
import render_classes
import render_component
import json
import result_loader

import openai
import tiktoken

ONLY_MISSING = True # only check if the fragment has not yet been processed

system_prompt = """list all the packages used in the following code.
Do not include any explanation or intro, just the list of packages as a json object. Only include a version number if it is required, otherwise presume the latest version.

good response:
{
    "x": "latest",
    "y": "1.0.0"
}

bad response:
[
    "x",
    "y"
]
"""
user_prompt = """{0}"""
term_prompt = "remember: only provide a object containing packages, no explanation is needed or intro is needed."


def generate_response(params, key):

    total_tokens = 0
    model = get_model_config('package_extractor', key)
    
    def reportTokens(prompt):
        encoding = tiktoken.encoding_for_model(model)
        # print number of tokens in light gray, with first 10 characters of prompt in green
        token_len = len(encoding.encode(prompt))
        print(
            "\033[37m"
            + str(token_len)
            + " tokens\033[0m"
            + " in prompt: "
            + "\033[92m"
            + prompt
            + "\033[0m"
        )
        return token_len

    # Set up your OpenAI API credentials
    openai.api_key = OPENAI_API_KEY

    messages = []
    prompt = system_prompt #.format(params['project_name'], params['project_description'],params['dev_stack'] , params['feature_title'])
    messages.append({"role": "system", "content": prompt})
    total_tokens += reportTokens(prompt)
    prompt = user_prompt.format(params['code'])
    messages.append({"role": "user", "content": prompt})
    total_tokens += reportTokens(prompt)
    if term_prompt:
        messages.append({"role": "assistant", "content": term_prompt})
        total_tokens += reportTokens(term_prompt)
    
    total_tokens += 25 
    if total_tokens > DEFAULT_MAX_TOKENS:
        total_tokens = DEFAULT_MAX_TOKENS
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": total_tokens,
        "temperature": 0,
    }

    # Send the API request
    keep_trying = True
    response = None
    while keep_trying:
        try:
            response = openai.ChatCompletion.create(**params)
            keep_trying = False
        except Exception as e:
            # e.g. when the API is too busy, we don't want to fail everything
            print("Failed to generate response (retrying in 30 sec). Error: ", e)
            sleep(30)
            print("Retrying...")

    # Get the reply from the API response
    if response:
        reply = response.choices[0]["message"]["content"] # type: ignore
        print("response: ", reply)
        return reply
    return None

def add_result(to_add, writer):
    writer.write(to_add + "\n")
    writer.flush()


def collect_response(title, response, writer):
    # get the first line in the component without the ## and the #
    add_result(f'# {title}', writer)
    add_result(response, writer)


def process_data(writer):
    for fragment in project.fragments:
        if fragment.content == '':
            continue
        if ONLY_MISSING and has_fragment(fragment.full_title):
            continue
        fragment_results = {}
        for file_path in render_classes.get_data(fragment.full_title):
            with open(file_path, "r") as reader:
                code = reader.read()
            params = {
                'code': code,
            }
            response = generate_response(params, fragment.full_title)
            if response:
                fragment_results[file_path] = response
        for file_path in render_component.get_data(fragment.full_title): 
            with open(file_path, "r") as reader:
                code = reader.read()
            params = {
                'code': code,
            }
            response = generate_response(params, fragment.full_title)
            if response:
                fragment_results[file_path] = response
        collect_response(fragment.full_title, json.dumps(fragment_results), writer)
                    


def main(prompt, classes, components, file=None):
    # read file from prompt if it ends in a .md filetype
    with open(prompt, "r") as promptfile:
        prompt = promptfile.read()

    print("loading project")

    # split the prompt into a toolbar, list of components and a list of services, based on the markdown headers
    project.split_standard(prompt)
    render_classes.load_results(classes)
    render_component.load_results(components)

    # save there result to a file while rendering.
    if file is None:
        file = 'output'
        
    file_name = file + "_packages.md"
    open_mode = 'w'
    if ONLY_MISSING:
        load_results(file_name)
        open_mode = 'a'

    print("rendering results")
    with open(file_name, open_mode) as writer:
        process_data(writer)
    
    print("done! check out the output file for the results!")


text_fragments = []  # the list of text fragments representing all the results that were rendered.


def load_results(filename, overwrite_file_name=None, overwrite=True):
    if not overwrite_file_name and overwrite:
        # modify the filename so that the filename without extension ends on _overwrite
        overwrite_file_name = filename.split('.')[0] + '_overwrite.' + filename.split('.')[1]
    result_loader.load(filename, text_fragments, True, overwrite_file_name)
        

def has_fragment(title):
    '''returns true if the title is in the list of fragments'''
    to_search = title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title == to_search:
            return True
    return False    

if __name__ == "__main__":

    # Check for arguments
    if len(sys.argv) < 4:
        # Still no? Then we can't continue
        print("Please provide a prompt and compressed version")
        sys.exit(1)
    
    else:
        # Set prompt to the first argument
        prompt = sys.argv[1]
        classes = sys.argv[2]
        components = sys.argv[3]

    # Pull everything else as normal
    file = sys.argv[4] if len(sys.argv) > 4 else None

    # Run the main function
    main(prompt, classes, components, file)
