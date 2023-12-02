import sys
import os
from time import sleep
from constants import get_model_config, DEFAULT_MAX_TOKENS, OPENAI_API_KEY
import project
import class_lister
import result_loader
import json
import primary_class
import class_descriptions

import openai
import tiktoken

ONLY_MISSING = True # only check if the fragment has not yet been processed

system_prompt = """Act as an ai software analyst.
It is your task to classify if the class '{0}', described as: '{1}', is declared in one of the given titles.
Only return no or the title it is declared in, do not include any explanation. Only return 1 title.

good response:
no

bad response:
The class 'X' is not declared in any of the titles."""
user_prompt = """titles:
{0}"""
term_prompt = """"""


def generate_response(params, key):

    total_tokens = 0
    model = get_model_config('declare_or_use_class', key)
    
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
    prompt = system_prompt.format(params['class'], params['class_description'] )
    messages.append({"role": "system", "content": prompt})
    total_tokens += reportTokens(prompt)
    prompt = user_prompt.format(params['titles'])
    messages.append({"role": "user", "content": prompt})
    total_tokens += reportTokens(prompt)
    if term_prompt:
        messages.append({"role": "assistant", "content": term_prompt})
        total_tokens += reportTokens(term_prompt)
    
    total_tokens += 20  # max result needs to be short
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


def add_result(to_add, result, writer):
    result.append(to_add)
    writer.write(to_add + "\n")
    writer.flush()


def collect_response(title, response, result, writer):
    # get the first line in the component without the ## and the #
    add_result(f'# {title}', result, writer)
    add_result(response, result, writer)


def process_data(writer):
    result = []
    
    titles = [fragment.full_title.split('#')[-1].strip() for fragment in class_lister.text_fragments]

    for fragment in project.fragments:
        if ONLY_MISSING and has_fragment(fragment.full_title):
            continue
        fragment_title = fragment.full_title.split('#')[-1].strip()
        classes = class_lister.get_classes(fragment.full_title)
        if len(classes) > 0:
            primary = primary_class.get_primary(fragment.full_title)
            response_dict = {}
            for item in classes:
                if item == primary:
                    response_dict[item] = 'declare'
                else:
                    description = class_descriptions.get_description(fragment.full_title, item)
                    other_titles = [title for title in titles if title != fragment_title]
                    params = {
                        'class': item,
                        'titles': other_titles,
                        'class_description': description
                    }
                    response = generate_response(params, fragment.full_title)
                    if response.lower() == 'no':
                        response = 'declare'
                    if response:
                        response_dict[item] = response
            collect_response(fragment.full_title, json.dumps(response_dict), result, writer)
                        
    return result
                    


def main(prompt, components_list, primary_list, class_descr, file=None):
    # read file from prompt if it ends in a .md filetype
    if prompt.endswith(".md"):
        with open(prompt, "r") as promptfile:
            prompt = promptfile.read()

    print("loading project")

    # split the prompt into a toolbar, list of components and a list of services, based on the markdown headers
    project.split_standard(prompt)
    class_lister.load_results(components_list)
    primary_class.load_results(primary_list)
    class_descriptions.load_results(class_descr)

    # save there result to a file while rendering.
    if file is None:
        file = 'output'
    
    file_name = file + "_declare_or_use_class.md"
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


def get_is_declared(full_title, class_name):
    """Returns true if the class is declared in the given text fragment"""
    to_search = full_title.lower().strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.full_title.lower().strip() == to_search:
            if class_name in fragment.data:
                return fragment.data[class_name] == 'declare'
    return False


def get_declared_in(full_title, name):
    """Returns the title of the text fragment where the class is declared in the given text fragment"""
    to_search = full_title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.full_title.strip() == to_search:
            if name in fragment.data:
                return fragment.data[name]
    return None


def get_all_declared(full_title):
    """Returns the list of classes that are declared in the given text fragment"""
    to_search = full_title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.full_title.strip() == to_search:
            return [name for name in fragment.data if fragment.data[name] == 'declare']
    return []


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
    if len(sys.argv) < 5:
        print("Please provide a prompt and a file containing the components to check")
        sys.exit(1)
    else:
        # Set prompt to the first argument
        prompt = sys.argv[1]
        components_list = sys.argv[2]
        primary_list = sys.argv[3]
        comp_descr = sys.argv[4]

    # Pull everything else as normal
    file = sys.argv[5] if len(sys.argv) > 5 else None

    # Run the main function
    main(prompt, components_list, primary_list, comp_descr, file)
