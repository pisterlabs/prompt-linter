from io import BytesIO
from typing import List, Optional, Union

import yaml
from copy import deepcopy

from yaml import SafeLoader

from .templates import csv_template, image_template, sc_template, standard_parameter_automation_prompt, standard_input_automation_prompt, jupyter_parameter_automation_prompt
from .templates import argparse_tags, click_tags, allowed_types, allowed_args, boolean_values, MAX_PARAMETERS, MAX_INPUTS

import openai
import re
from os import environ


PARSE_WITH_CHATGPT_MODE = 'chatgpt_parse'
PARSE_MANUALLY_MODE = 'substring_parse'


def _parse_input_python(file: BytesIO):
    byte_lines = file.readlines()
    line_array = []

    argparse_flag = False
    click_flag = False

    for byte_line in byte_lines:
        # example of str_line: 'b'import os\r\n''
        str_line = str(byte_line)
        line = str_line.replace("b'", "").replace("\\r", "").replace("\\n", "")[:-1]
        # TODO add if line ends with coma - add next line to this line.
        # Strips the newline character
        line_array.append(line.strip())
        if any(tag in line for tag in argparse_tags):
            argparse_flag = True
        if any(tag in line for tag in click_tags):
            click_flag = True

    if argparse_flag and not click_flag:
        return line_array, "argparse"
    elif click_flag and not argparse_flag:
        return line_array, "click"
    else:
        return None, None
        
        
def _dict_from_args(filelines: List[str], library: str):
    if library == 'argparse':
        arg_command = '.add_argument('
    else:
        arg_command = '.option('
        
    parameter_dict = {}
    for line in filelines:
        if arg_command not in line:
            continue
        arg_string = line.split(arg_command)[1]
        arguments = arg_string.split(',')
        argname = arguments[0].split('--')[1].strip().strip("'")
        parameter_dict[argname] = {}

        for argument in arguments[1:]:
            argument_split = argument.strip().split('=')
            arg_value = ''.join(argument_split[1:])
            if len(arg_value.strip("'")) == 0:
                continue
            if arg_value.count('(') < arg_value.count(')'):
                arg_value = arg_value.rstrip(')')
            if arg_value[0] == "'" and arg_value[-1] == "'":
                arg_value = arg_value[1:-1]
            if len(argument_split) > 1:
                parameter_dict[argname][argument_split[0]] = arg_value
                    
    return parameter_dict


def _stages_from_scripts(files_data: List[dict]):
    stages = {}
    for file_details in files_data:
        file, file_type, file_name = file_details['file'], file_details['file_type'], file_details['file_name']
        # separate file name from file extension
        file_name = file_name.split('/')[-1].split(f".{file_type}")[0]
        stages[file_name] = {'file': file}
    return json_to_yaml(stages)


def _is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    

def _attempt_numeric(string, ntype):
    if ntype == 'int':
        try:
            return int(string)
        except ValueError:
            return string
    if ntype == 'float':
        try:
            return float(string)
        except ValueError:
            return string        
        

def _format_argparse_parameters(input_parameters):
    parameters = deepcopy(input_parameters)
    
    for key, subdict in parameters.items():
        # inferring / correcting type
        if 'type' in subdict:
            if subdict['type'] not in allowed_types:
                subdict['type'] = 'str'
        elif 'is_flag' in subdict:
            subdict['type'] = 'boolean'
            if 'default' not in subdict:
                subdict['default'] = False
        elif 'default' in subdict:
            if subdict['default'] in boolean_values:
                subdict['type'] = 'boolean'
            elif subdict['default'].isdigit():
                subdict['type'] = 'int'
            elif _is_float(subdict['default']):
                subdict['type'] = 'float'
            else:
                subdict['type'] = 'str'
        # adjusting naming conventions
        if 'help' in subdict:
            subdict['tooltip'] = subdict.pop('help')
        if 'choices' in subdict:
            subdict['options'] = subdict.pop('choices')
        if 'min' in subdict:
            subdict['min_value'] = subdict.pop('min')
        if 'max' in subdict:
            subdict['max_value'] = subdict.pop('max')
        # adjusting number formatting
        if 'type' in subdict:
            if subdict['type'] in ['int', 'float']:
                if 'default' in subdict:
                    subdict['default'] = _attempt_numeric(subdict['default'], subdict['type'])
                if 'min_value' in subdict:
                    subdict['min_value'] = _attempt_numeric(subdict['min_value'], subdict['type'])
                if 'max_value' in subdict:
                    subdict['max_value'] = _attempt_numeric(subdict['max_value'], subdict['type'])
                if 'increment' in subdict:
                    subdict['increment'] = _attempt_numeric(subdict['increment'], subdict['type'])
        # remove excess quotes from tooltip
        if 'tooltip' in subdict:
            raw_tip = subdict['tooltip'].strip("'")
            subdict['tooltip'] = str(raw_tip)
        # removing unknown arguments
        del_list = []
        for argkey in subdict.keys():
            if argkey not in allowed_args:
                del_list.append(argkey)
        [subdict.pop(argkey) for argkey in del_list]
    return parameters


def _prune_script(script_text):
    substrings = ["@click.option(", "parser.add_argument("]
    new_text = ""
    lines = list(set(script_text.splitlines()))

    for line in lines:
        if any(line.find(substring)>=0 for substring in substrings):
            new_text += line + "\n"
    
    return new_text


def _prune_jupyter(script_text):
    startstrings = ["#", "'''", "import", "from"]
    endstrings = ["'''"]
    substrings = ["print", "ipython"]
    keep = ["="] #,"def ","class "]
    new_text = ""
    lines = list(set(script_text.splitlines()))
    for line in lines:
        if any(line.lstrip().startswith(substring) for substring in startstrings):
            continue
        if any(line.rstrip().endswith(substring) for substring in endstrings):
            continue
        if any(substring in line for substring in substrings):
            continue
        if not any(substring in line for substring in keep):
            continue
        new_text += line + "\n"
    return new_text


def _parse_input_python_v2(file: BytesIO, file_type: str = 'py', verbose: bool = False):
    script_text = file.getvalue().decode('ASCII')
    if file_type=='py':
        stripped_script = _prune_script(script_text)
    elif file_type=='ipynb':
        stripped_script = _prune_jupyter(script_text)
    if verbose:
        line_count1 = len(script_text.splitlines())
        line_count2 = len(stripped_script.splitlines())
        print(f"Length of full script is {line_count1}, and the stripped script is {line_count2}")
    return stripped_script


def _parse_multiple_files(files_data: List[dict], verbose=False):
    list_contents = []
    ipynb_detected = False
    for file_details in files_data:
        list_contents.append(_parse_input_python_v2(file_details['file'], file_details['file_type'], verbose))
        if file_details['file_type'] == 'ipynb':
            ipynb_detected = True        
    delimiter = '\n'
    result = delimiter.join(list_contents)
    return result, ipynb_detected


def openai_chat_completion(prompt, file_contents, max_token=50, outputs=1, temperature=0.75, model="gpt-4-0613"):
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": file_contents}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_token,
        n=outputs,
        # The range of the sampling temperature is 0 to 2.
        # lower values like 0.2=more random, higher values like 0.8=more focused and deterministic
        temperature=temperature
    )
    if outputs == 1:
        response_text = response['choices'][0]['message']['content']
    else:
        response_text = []
        for i in range(outputs):
            response_text.append(response['choices'][i]['message']['content'])
            
    return response_text


def _extract_yaml(output):
    # sometimes chatgpt returns yaml+an explanation of the yaml above and below. This keeps only the yaml
    matches = re.findall(r"```(.*?)```", output, re.DOTALL)
    if matches:
        return matches[0].replace('yaml', '')
    return output


def is_invalid_yaml(text):
    try:
        yaml.safe_load(text)
        return False
    except yaml.YAMLError as exc:
        return True
    
    
def substring_parse_inputs(parameters_yaml: str) -> str:
    parameters = yaml_to_json(parameters_yaml)
    input_settings = {}
    # AttributeError: 'str' object has no attribute 'values'
    for parameter_dict in parameters.values():
        if not all(k in parameter_dict.keys() for k in ("type", "default")):
            continue
        elif not parameter_dict['type'] in ['path', 'str']:
            continue
        else:
            file_split = parameter_dict['default'].split('.')
            # if a path, then usually has two parts
            if len(file_split) == 2:
                filename = file_split[0]
                fileext = file_split[1]
                if fileext in csv_template['allowedFormats']['fileExtensions']:
                    input_template = csv_template.copy()
                    input_settings[filename] = input_template
                elif fileext in image_template['allowedFormats']['fileExtensions']:
                    input_template = image_template.copy()
                    input_settings[filename] = input_template
                elif fileext in sc_template['allowedFormats']['fileExtensions']:
                    input_template = sc_template.copy()
                    input_settings[filename] = input_template

    return json_to_yaml(input_settings)


def validate_multiple_outputs(outputs):
    valid_outputs = []
    for output in outputs:
        valid = True
        # check has not returned directories or checkpoints as input files
        if any(substring in output for substring in ['directory', 'Directory', 'checkpoint', 'Checkpoint']):
            valid = False
        if is_invalid_yaml(output):
            output = _extract_yaml(output)
            if is_invalid_yaml(output):    
                valid = False
        if valid:
            valid_outputs.append(output)
    return valid_outputs


#input is a multiline string, formatted as yaml
def _prune_yaml(yaml_str: str, count: int):
    data = yaml.safe_load(yaml_str)
    top_keys = list(data.keys())[:count]
    new_data = {key: data[key] for key in top_keys}
    pruned_yaml = yaml.dump(new_data, default_flow_style=False, explicit_start=True)
    return pruned_yaml


def chatgpt_parse_parameters(file_contents, ipynb_detected):
    openai.api_key = environ.get("OPENAI_KEY")
    if ipynb_detected:
        parameters = openai_chat_completion(jupyter_parameter_automation_prompt, file_contents, max_token=3000, outputs=1)
    else:
        parameters = openai_chat_completion(standard_parameter_automation_prompt, file_contents, max_token=3000, outputs=1)
    if is_invalid_yaml(parameters):
        formatted_parameters = _extract_yaml(parameters)
        pruned_yaml = _prune_yaml(formatted_parameters, MAX_PARAMETERS)
        if is_invalid_yaml(pruned_yaml):
            raise ValueError('Invalid YAML format for parameters.')
        else:
            return pruned_yaml
    else:
        pruned_yaml = _prune_yaml(parameters, MAX_PARAMETERS)
        return pruned_yaml
    

def chatgpt_parse_inputs(file_contents):
    openai.api_key = environ.get("OPENAI_KEY")
    input_options = openai_chat_completion(standard_input_automation_prompt, file_contents, max_token=300,
                                           outputs=10, temperature=0.9)
    valid_options = validate_multiple_outputs(input_options)
    if len(valid_options) > 0:
        input_settings = _prune_yaml(valid_options[0], MAX_INPUTS)
    else:
        raise ValueError('Invalid YAML format for inputs.')
    return input_settings


def json_to_yaml(json_value: Optional[dict]) -> str:
    return yaml.dump(json_value) if json_value else '\n'


def yaml_to_json(str_value: str) -> Optional[dict]:
    dict_value = yaml.load(str_value, Loader=SafeLoader) if str_value else None
    return dict_value


def substring_parse_parameters(files_data: List[dict]) -> str:
    parameters = {}
    library_found = False
    
    for file_details in files_data:
        file_lines, argument_parsing_library = _parse_input_python(file_details['file'])
        if argument_parsing_library is not None:
            library_found = True
            new_parameters = _dict_from_args(file_lines, argument_parsing_library)
            parameters = {**parameters, **new_parameters}
    formatted_parameters = _format_argparse_parameters(parameters) if library_found else parameters
    return json_to_yaml(formatted_parameters)


def _validate_param_dict(files_data: List[dict]):
    for file_dict in files_data:
        if 'file' not in file_dict:
            raise ValueError('file not found in input dict')
        elif not isinstance(file_dict['file'], (bytes, bytearray, BytesIO)):
            raise ValueError('file is not a BytesIO object')
        if 'file_type' not in file_dict:
            raise ValueError('file_type not found in input dict')
        if 'file_name' not in file_dict:
            raise ValueError('file_name not found in input dict')
            
    
def parameters_yaml_from_args(files_data: List[dict],
                              method: Union[PARSE_WITH_CHATGPT_MODE, PARSE_MANUALLY_MODE] = PARSE_WITH_CHATGPT_MODE) \
        -> (str, str, str):
    '''file_dict configuration:
    file_name as keys,
    file: BytesIO,
    file_type: str = 'py'
    example: file_dict = {fileone: {file: BytesIO, file_type: 'py'}, filetwo: {file: BytesIO, file_type: 'py'}}'''
    _validate_param_dict(files_data)

    if method == PARSE_WITH_CHATGPT_MODE:
        file_contents, ipynb_detected = _parse_multiple_files(files_data=files_data, verbose=False)
        try:
            formatted_parameters = chatgpt_parse_parameters(file_contents, ipynb_detected)
        except Exception as e:
            formatted_parameters = substring_parse_parameters(files_data)
        try:
            input_settings = chatgpt_parse_inputs(file_contents)
        except Exception as e:
            input_settings = substring_parse_inputs(formatted_parameters)

    elif method == PARSE_MANUALLY_MODE:
        formatted_parameters = substring_parse_parameters(files_data)
        input_settings = substring_parse_inputs(formatted_parameters)
    
    stages = _stages_from_scripts(files_data)
    
    # output settings not covered
    return stages, formatted_parameters, input_settings
