# lcreates a json list of all the properties declared for by a component
import sys
from time import sleep
from constants import get_model_config, DEFAULT_MAX_TOKENS, OPENAI_API_KEY
import project
import declare_or_use_comp_classifier
import component_descriptions_exact
import json
import result_loader

import openai
import tiktoken

ONLY_MISSING = True # only check if the fragment has not yet been processed

system_prompt = """list all features that the {0} component declares consumers of the {0} component should provide as property values for the {0}.
Return the result as a json object of key-value pairs where the value is a short description. Do not include any introduction or explanation. Return an empty object if nothing is found."""
user_prompt = """{0}"""
term_prompt = """"""


def generate_response(params, key):

    total_tokens = 0
    model = get_model_config('list_component_props', key)
    
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
    prompt = system_prompt.format(params['component_name'] )
    messages.append({"role": "system", "content": prompt})
    total_tokens += reportTokens(prompt)
    prompt = user_prompt.format(params['component_description'])
    messages.append({"role": "user", "content": prompt})
    total_tokens += reportTokens(prompt)
    if term_prompt:
        prompt = term_prompt.format(params['component_name'])
        messages.append({"role": "assistant", "content": prompt})
        total_tokens += reportTokens(prompt)
    
    total_tokens += 120  
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
    add_result(title, result, writer)
    add_result(response, result, writer)


def process_data(writer):
    result = []


    for to_check in declare_or_use_comp_classifier.text_fragments:
        if ONLY_MISSING and has_fragment(to_check.full_title):
            continue
        components = to_check.data
        if not components:
            continue
        results = {}
        for component_name, value in components.items():
            if value == 'declare':
                description = component_descriptions_exact.get_description(to_check.full_title, component_name)
                params = {
                    'component_name': component_name,
                    'component_description': description,
                }
                response = generate_response(params, to_check.full_title)
                sleep(1) # wait a bit to prevent the api from getting overloaded
                if response:
                    try:
                        results[component_name] = json.loads(response)
                    except Exception as e:
                        print("Failed to parse response: ", e)
                        print("Response: ", response)
                        results[component_name] = {}
                else:
                    results[component_name] = {}
                    
        collect_response(to_check.full_title, json.dumps(results), result, writer)
    return result
                    


def main(prompt, class_list, descriptions, file=None):
    # read file from prompt if it ends in a .md filetype
    if prompt.endswith(".md"):
        with open(prompt, "r") as promptfile:
            prompt = promptfile.read()

    print("loading project")

    # split the prompt into a toolbar, list of components and a list of services, based on the markdown headers
    project.split_standard(prompt)
    declare_or_use_comp_classifier.load_results(class_list)
    component_descriptions_exact.load_results(descriptions)

    if file is None:
        file = 'output'

    filename = file + "_component_props.md"
    open_mode = 'w'
    if ONLY_MISSING:
        load_results(filename)
        open_mode = 'a'

    print("rendering results")

    # save there result to a file while rendering.
    with open(filename, open_mode) as writer:
        process_data(writer)
    
    print("done! check out the output file for the results!")


text_fragments = []  # the list of text fragments representing all the results that were rendered.

def load_results(filename, overwrite_file_name=None, overwrite=True):
    if not overwrite_file_name and overwrite:
        # modify the filename so that the filename without extension ends on _overwrite
        overwrite_file_name = filename.split('.')[0] + '_overwrite.' + filename.split('.')[1]
    result_loader.load(filename, text_fragments, True, overwrite_file_name)


def get_data(title):
    '''returns the list of components for the given title'''
    to_search = title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title == to_search:
            return fragment.data or {}
    return {}   

def has_fragment(title):
    '''returns true if the title is in the list of fragments'''
    to_search = title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title == to_search:
            return True
    return False

def get_props(title, comp):
    '''the data found for the given component'''
    data = get_data(title)
    if not comp in data:
        return []
    section = data[comp]
    if not section:
        return []
    return section

if __name__ == "__main__":

    # Check for arguments
    if len(sys.argv) < 4:
        print("Please provide a prompt")
        sys.exit(1)
    else:
        # Set prompt to the first argument
        prompt = sys.argv[1]
        class_list = sys.argv[2]
        descriptions = sys.argv[3]

    # Pull everything else as normal
    file = sys.argv[4] if len(sys.argv) > 4 else None

    # Run the main function
    main(prompt, class_list, descriptions, file)
