import sys
import os
from time import sleep
from constants import  get_model_config, DEFAULT_MAX_TOKENS, OPENAI_API_KEY, MAX_TOKENS
import project
import class_lister
import declare_or_use_class_classifier
import get_if_service_is_singleton
import get_interface_parts
import get_interface_parts_usage
import resolve_class_imports
import list_service_usage
import primary_class
import constant_lister
import json
import result_loader

import openai
import tiktoken

ONLY_MISSING = True # only check if the fragment has not yet been processed

system_prompt = """Act as a full-stack ai software developer.
It is your task to write all the code for the class '{0}'

use the following development stack:
{1}
"""
user_prompt = """The class '{0}' is described as follows:
{1}
{2}
{3}"""
term_prompt = """
Use small functions.
A file always contains the definition for 1 component, service or object, no more.
Add documentation to your code.
Only write valid code.
Fully implement all pseudo code.
Do not include any intro or explanation, only write code"""


def generate_response(params, key):

    total_tokens = 0
    model = get_model_config('render_class', key)
    
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
    prompt = system_prompt.format(params['class'], params['dev_stack'], params['imports'] ) + term_prompt
    messages.append({"role": "system", "content": prompt})
    total_tokens += reportTokens(prompt)
    imports_txt = ''
    if params['imports']:
        imports_txt = f'imports (only include the imports that are used in the code):\n{params["imports"]}'
    prompt = user_prompt.format(params['class'], params['feature_description'].strip(), params['interface_parts'], imports_txt)
    prompt = prompt.strip()
    messages.append({"role": "user", "content": prompt})
    total_tokens += reportTokens(prompt)
    
    total_tokens = (total_tokens * 2) + DEFAULT_MAX_TOKENS # code needs max allowed
    if total_tokens > MAX_TOKENS:
        total_tokens = MAX_TOKENS
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


def collect_file_list(title, file_names, writer):
    names_str = json.dumps(file_names)
    writer.write(f'# {title}\n')
    writer.write(f'{names_str}\n')
    writer.flush()


def get_file_path(title, root_path):
    file_name = title.replace(" > ", "_")
    file_path = os.path.join(root_path, file_name.replace(" ", "_") + ".js")
    return file_path


def collect_response(title, response, root_path):
    file_name = title.replace(" > ", "_")
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    file_path = os.path.join(root_path, file_name.replace(" ", "_") + ".js")
    with open(file_path, "w") as writer:
        writer.write(response)
    return file_path


def get_to_render_and_used(title, classes):
    used = []
    to_render = []
    for cl in classes:
        is_declare = declare_or_use_class_classifier.get_is_declared(title, cl)
        if not is_declare:
            used.append(cl)
        else:
            to_render.append(cl)
    return to_render, used


def get_description_and_interface_parts(fragment, cl, to_render):
    if len(to_render) > 1:
        print('more than 1 class to render, this is not fully supported yet')
    content = constant_lister.get_fragment(fragment.full_title, fragment.content)
    feature_description = content
    interface_parts = get_interface_parts.get_interface_parts(fragment.full_title, cl) # things this component needs to implement
    if interface_parts:
        to_join = []
        for key, value in interface_parts.items():
            to_join.append(f'{key}: {value}')
        interface_parts = f'\nMake certain that {cl} has:\n- ' + '\n- '.join(to_join) + '\n'
    if not interface_parts:
        interface_parts = ''
    return feature_description, interface_parts


def format_interface(interface_def):
    result = ''
    for key, value in interface_def.items():
        result += f'- {key}: {value}\n'
    return result


def get_interface_parts_of_others(fragment, component):
    """
    goes over all the services that are imported by the component and builds up the interface from them so that 
    the entire application uses the same interface for the same service
    """
    result = ''
    imports = resolve_class_imports.get_data(fragment.full_title)
    if not imports:
        return result
    items = imports.get(component)
    if items:
        print('todo: add class-description-exact converter')
        class_desc = fragment.content # component_descriptions_exact.get_description(fragment.full_title, component)
        for import_def in items:
            service = import_def['service']
            service_loc = import_def['service_loc']
            interface_def = get_interface_parts.get_interface_parts(service_loc, service)
            if interface_def:
                comp_and_global_service_desc = class_desc
                # if there would be global descriptions for services, add them here to comp_and_global_service_desc
                interface_def = format_interface(interface_def)
                interface = get_interface_parts_usage.list_used_interface_parts(service, service_loc, interface_def, comp_and_global_service_desc, fragment.full_title)
                if interface:
                    interface_list = []
                    interface_parts_def = get_interface_parts.get_interface_parts(service_loc, service)
                    for key, value in interface.items():
                        if key in interface_parts_def: # prefer the description that is extracted with get_interface_parts, gives better results (less chatter)
                            description = interface_parts_def[key]
                        else:
                            description = value
                        interface_list.append(f'{key}: {description}')
                    result += f'\n{service} has the following interface:\n- ' + '\n- '.join(interface_list) + '\n'
    return result


def get_import_service_line(import_def, cur_path):
    service = import_def['service']
    service_path = import_def['path']
    service_path = os.path.relpath(service_path, cur_path)
    is_global = get_if_service_is_singleton.is_singleton(import_def['service_loc'], service)
    if is_global:
        service_txt = "global object"
        service = service.lower()
    else:
        service_txt = "service"
    return f"The {service_txt} {service} can be imported from {service_path} (exported as default)\n"


def get_all_imports(cl_to_render, full_title, cur_path, root_path):
    imports_txt = ''
    has_constants = constant_lister.has_constants(full_title)
    if has_constants:
        rel_path = os.path.relpath(constant_lister.get_resource_filename(root_path), os.path.join(root_path, cur_path))
        imports_txt += f"The const 'resources' can be imported from {rel_path}\n"
    cur_path = cur_path.strip()
    imports = resolve_class_imports.get_data(full_title)
    if not imports:
        return imports_txt
    primary = primary_class.get_primary(full_title) # only the primary component imports other components declared in the same fragment. this is to prevent confusion from gpt
    already_imported = {}
    for comp, items in imports.items():
        if comp == cl_to_render:
            for import_def in items:
                if import_def['service'] in already_imported:
                    continue
                imports_txt += get_import_service_line(import_def, cur_path)
                already_imported[import_def['service']] = True
        else:
            is_declare = declare_or_use_class_classifier.get_is_declared(full_title, comp)
            if is_declare:
                # only import components from the same folder if we are rendering the primary component, which uses the children
                if primary == cl_to_render: # this is to prevent confusion from gpt, otherwise it starts using the parent component in the children
                    # another component declared in the same fragment, so import from local path
                    imports_txt += f"The class {comp} can be imported from ./{comp}\n"
            else:
                if comp in already_imported:
                    continue
                rel_path = os.path.relpath(items.strip(), cur_path)
                imports_txt += f"The class {comp} can be imported from {rel_path}\n"
                already_imported[comp] = True
    return imports_txt  


def render_class(cl, fragment, to_render, root_path, file_names):
    # calculate the path to the files we will generate cause we need it to get the import paths of the locally declared components
    title_to_path = fragment.full_title.replace(":", "").replace('?', '').replace('!', '')
    path_items = title_to_path.split(" > ")
    path_items = [part.replace(" ", "_") for part in path_items]
    path_items[0] = 'src' # the first item is the project name, we need to replace it with src so that the code gets rendered nicely
    path_section = os.path.join(root_path, *path_items)
    relative_path = os.path.join(*path_items)

    feature_description, interface_parts = get_description_and_interface_parts(fragment, cl, to_render)
    others_interface_parts = get_interface_parts_of_others(fragment, cl)
    # global_features = get_global_features(fragment.full_title, cl)

    if interface_parts and others_interface_parts:
        interface_parts += '\n'
    interface_parts += others_interface_parts

    params = {
        'class': cl,
        'feature_title': fragment.title,
        'feature_description': feature_description,
        'dev_stack': project.fragments[1].content,
        'imports': get_all_imports(cl, fragment.full_title, relative_path, root_path),
        'interface_parts': interface_parts,
        # 'global_features': global_features,
    }
    response = generate_response(params, fragment.full_title)
    if response:
        # remove the code block markdown, the 3.5 version wants to add it itself
        response = response.strip() # need to remove the newline at the end
        if response.startswith("```javascript"):
            response = response[len("```javascript"):]
        if response.endswith("```"):
            response = response[:-len("```")]
        file_name = collect_response(cl, response, path_section)
        file_names.append(file_name)
        return response
    

def extract_service_interface_parts(code, fragment, rendered_comp):
    """
    extract the interface parts from the code and save them in the interface parts file
    """
    imports = resolve_class_imports.get_data(fragment.full_title)
    if imports:
        items = imports.get(rendered_comp)
        if items:
            for import_def in items:
                service = import_def['service']
                service_loc = import_def['service_loc']
                get_interface_parts.extract_interface_parts_for(service, service_loc, code, fragment.full_title)


def extract_used_comp_interface_parts(code, fragment, rendered_comp, used_classes):
    for cl in used_classes:
        cl_loc = declare_or_use_class_classifier.get_declared_in(fragment.full_title, cl)
        get_interface_parts.extract_interface_parts_for(cl, cl_loc, code, fragment.full_title)



def process_data(root_path, writer):
    for fragment in project.fragments:
        if ONLY_MISSING and has_fragment(fragment.full_title):
            continue
        classes = class_lister.get_classes(fragment.full_title)
        if len(classes) > 0:
            file_names = [] # keep track of the file names generated for this fragment, so we can save it in the markdown file
            to_render, used = get_to_render_and_used(fragment.full_title, classes)
            primary = primary_class.get_primary(fragment.full_title)
            if not primary:
                raise Exception('no primary found for ', fragment.full_title)
            # we presume that only the primary class is currently using the sub classes and non of the subs use each other.
            # to save some cost & time. this means don't put too complex stuff in a single fragment
            primary_code = render_class(primary, fragment, to_render, root_path, file_names)
            extract_service_interface_parts(primary_code, fragment, primary)
            extract_used_comp_interface_parts(primary_code, fragment, primary, used)

            non_primary = [c for c in to_render if c != primary]
            for cl in non_primary:
                get_interface_parts.extract_interface_parts_for(cl, fragment.full_title, primary_code, fragment.full_title)
                code = render_class(cl, fragment, to_render, root_path, file_names)
                extract_service_interface_parts(code, fragment, cl)
            if file_names:
                collect_file_list(fragment.full_title, file_names, writer)
                    


def main(prompt, components_list, declare_or_use_list, expansions, interface_parts, interface_parts_usage, imports, primary, singleton, constants, root_path=None, file=None):
    # read file from prompt if it ends in a .md filetype
    if prompt.endswith(".md"):
        with open(prompt, "r") as promptfile:
            prompt = promptfile.read()

    print("loading project")

    # split the prompt into a toolbar, list of components and a list of services, based on the markdown headers
    project.split_standard(prompt)
    class_lister.load_results(components_list)
    declare_or_use_class_classifier.load_results(declare_or_use_list)
    list_service_usage.load_results(expansions)
    get_interface_parts.load_results(interface_parts)
    get_interface_parts_usage.load_results(interface_parts_usage)
    resolve_class_imports.load_results(imports)
    primary_class.load_results(primary)
    get_if_service_is_singleton.load_results(singleton)
    constant_lister.load_results(constants)

    # save there result to a file while rendering.
    if file is None:
        file = 'output'
        
    open_mode = 'w'
    if ONLY_MISSING:
        load_results(file + "_class_files.md")
        open_mode = 'a'

    print("rendering results")

    # save there result to a file while rendering.
    if root_path is None:
        root_path = './'
    
    
    with open(file + "_class_files.md", open_mode) as writer:
        process_data(root_path, writer)
    
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


def get_data(title):
    '''returns the list of components for the given title'''
    to_search = title.lower().strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title.lower() == to_search:
            return fragment.data or []
    return []    


if __name__ == "__main__":

    # Check for arguments
    if len(sys.argv) < 12:
        print("Please provide a prompt and a file containing the components to check")
        sys.exit(1)
    else:
        # Set prompt to the first argument
        prompt = sys.argv[1]
        components_list = sys.argv[2]
        declare_or_use_list = sys.argv[3]
        expansions = sys.argv[4]
        interface_parts = sys.argv[5]
        interface_parts_usage = sys.argv[6]
        imports = sys.argv[7]
        primary = sys.argv[8]
        singleton = sys.argv[9] 
        constants = sys.argv[10] 

    # Pull everything else as normal
    folder = sys.argv[11] if len(sys.argv) > 11 else None
    file = sys.argv[12] if len(sys.argv) > 12 else None

    # Run the main function
    main(prompt, components_list, declare_or_use_list, expansions, interface_parts, interface_parts_usage,
         imports, primary, singleton, constants, folder, file)
