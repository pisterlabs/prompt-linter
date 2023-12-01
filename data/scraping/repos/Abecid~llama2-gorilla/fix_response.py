import json
import csv
from tqdm import tqdm
from datetime import datetime
import re
import ast
import os

from prompts import template
from prompts import example as prompt_examples
import openai_api
import check

conda = "openai_conda-2023Jun12.json"
linux = "openai_linux-2023Jun13.json"
rapidAPI = "rapidAPI-api_09_30_gpt_3_5_turbo.json"
aws = "aws-cli-2023_09_29_gpt_3_5_turbo.json"
output_aws = "aws-cli-2023_09_29_gpt_3_5_turbo_10_08_00_00_cleaned.json"

filenames = ['aws-cli-2023_09_29_gpt_3_5_turbo_10_08_00_00_cleaned_10_12_20_48_additional_fixed_fixed.json',
             'openai_gcloud-2023Jun13_fixed_10_19_fixed_fixed.json',
             'openai_github-2023Jun13_fixed_10_19_fixed_fixed.json',
             'rapidAPI-api_09_30_gpt_3_5_turbo_10_06_18_53_cleaned_additional_fixed_fixedArguments_fixed.json',
             'rapidAPI-api_09_30_gpt_3_5_turbo_10_06_18_53_cleaned_fixed_fixed.json'
             ]

inputs = [conda, linux]

model = "gpt-3.5-turbo"
# model = "gpt-4"

model_clean_name = model.replace(".", "-").replace("/", "_").replace(" ", "_").replace("-", "_")

OpenAI_API = openai_api.OpenAI_API(model=model)

# Function to convert string representation of dict to actual dict
def string_to_dict(s):
    s = "{" + s + "}"
    return json.loads(s.replace("\\", "").replace("'", "\""))

def get_correct_model_answer_python(model_answer):
    prompt = template.fix_response_to_python.replace("<<<EXAMPLE_API_CALL>>>", model_answer).replace("<<<EXAMPLES>>>", prompt_examples.FIX_RESPONSE_TO_PYTHON)
            
    response = OpenAI_API.chatgpt(prompt).strip()
    return response

def fix_model_name(s):
    last_dot_index = s.rfind('.')
    open_parenthesis_index = s.find('(', last_dot_index)

    function_name = s[:open_parenthesis_index]
    return function_name

def fix_query(query):
    if '\ngcloud' in query:
        query = query[:query.find('\ngcloud')].strip()
    if '\ngit' in query:
        query = query[:query.find('\ngit')].strip()
    return query

def fix_gcloud_model_answer(model_answer):
    if len(model_answer) == 0:
        return False
    i = 0
    while i < 3:
        if i == 1:
            model_answer.replace('-', '_')
            if '\ngcloud' in model_answer:
                model_answer = model_answer[model_answer.find('\ngcloud')+1:].split('\n')[0].strip()
            if '\ngit' in model_answer:
                model_answer = model_answer[model_answer.find('\ngit')+1:].split('\n')[0].strip()
        try:
            ast.parse(model_answer)
            if model_answer.find("import requests") == -1 and model_answer.find("\nprint") == -1 and model_answer.find("\n") == -1:
                return model_answer
                break  # Exit loop if the parsing is successful
            else:
                model_answer = get_correct_model_answer_python(model_answer)
                i += 1
        except SyntaxError:
            if (('gcloud.' not in model_answer) and ('git.' not in model_answer)) and '(' not in model_answer and ')' not in model_answer:
                return False
            if i == 0:
                model_answer.replace('-', '_')
                if '\ngcloud' in model_answer:
                    model_answer = model_answer[model_answer.find('\ngcloud')+1:].split('\n')[0].strip()
                i += 1
                continue
            if i == 1:
                print(f"Model Answer failed to parse: {model_answer}")
            model_answer = get_correct_model_answer_python(model_answer)
            i += 1
    return False
    return model_answer

def fix_model_names(input_filepath):
    with open(input_filepath, 'r') as file:
        data = json.load(file)
    for index, object in enumerate(tqdm(data)):
        try:
            object['original']['api_name_original'] = object['original']['api_name']
            model_answer = object["model_answer"]
            model_name = fix_model_name(model_answer)
            object['original']['api_name'] = model_name
        except Exception as e:
            print(f"Error: {e}")
            continue
    new_input_filepath = input_filepath.replace(".json", "_fixed.json")
    with open(f'{new_input_filepath}', 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)

def fix_gcloud(input_filepath, max=-1):
    new_data = []
    with open(input_filepath, 'r') as file:
        data = json.load(file)
    for index, object in enumerate(tqdm(data)):
        if max>0 and index >= max:
            break
        try:
            model_answer = object["model_answer"]
            model_answer = fix_gcloud_model_answer(model_answer)
            object["query"] = fix_query(object["query"])
            if model_answer is False:
                continue
            object['model_answer'] = model_answer
            new_data.append(object)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Model Answer: {model_answer}")
            continue
    new_input_filepath = input_filepath.replace(".json", "_fixed.json")
    with open(f'{new_input_filepath}', 'w') as jsonfile:
        json.dump(new_data, jsonfile, indent=4)
    

def fix_model_answer(model_answer):
    if len(model_answer) == 0:
        return False
    user_query_start = model_answer.find("User query")
    arguments_start = model_answer.find("<Arguments>")
    import_exists = model_answer.find("import requests")
    print_exists = model_answer.find("\nprint")
    newline_exists = model_answer.find("\n")
    if user_query_start != -1:
        model_answer = model_answer[:user_query_start].strip()
    if arguments_start != -1:
        model_answer = model_answer[:arguments_start].strip()
    if import_exists != -1:
        model_answer = model_answer.replace('import requests', '').strip()
    # last_par = model_answer.find(")")
    i = 0
    while i < 3:
        try:
            ast.parse(model_answer)
            if model_answer.find("import requests") == -1 and model_answer.find("\nprint") == -1 and model_answer.find("\n") == -1:
                break  # Exit loop if the parsing is successful
            else:
                model_answer = get_correct_model_answer_python(model_answer)
                i += 1
        except SyntaxError:
            if i == 1:
                print(f"Model Answer failed to parse: {model_answer}")
            model_answer = get_correct_model_answer_python(model_answer)
            i += 1
    # if last_par != -1 or ast.parse(model_answer) is not None:
    #     i = 0
    #     while True and i < 3:
    #         if i >= 1:
    #             print(f"Model Answer failed to parse {model_answer}")
    #         # model_answer = model_answer[:last_par+1].strip()
    #         model_answer = get_correct_model_answer_python(model_answer)
    #         if ast.parse(model_answer) is not None:
    #             break
    #         i += 1
            
    return model_answer
            
    
    # Check if empty
    # Check if last par is not the last character
    # Check AST Valid Python Code : ast.parse(model_answer)
    
def fix_rapidapi(input_filepath):
    new_data = []
    with open(input_filepath, 'r') as file:
        data = json.load(file)
    for object in tqdm(data):
        try:
            model_answer = object["model_answer"]
            model_answer = fix_model_answer(model_answer)
            if model_answer is False:
                continue
            object['model_answer'] = model_answer
            new_data.append(object)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Model Answer: {model_answer}")
            continue
    new_input_filepath = input_filepath.replace(".json", "_fixed.json")
    with open(f'{new_input_filepath}', 'w') as jsonfile:
        json.dump(new_data, jsonfile, indent=4)


def extract_arguments(s):
    url_pattern = r"requests\.get\(\"(.*?)\""
    headers_pattern = r"headers=\{(.*?)\}"
    params_pattern = r"params=\{(.*?)\}"

    url_match = re.search(url_pattern, s)
    headers_match = re.search(headers_pattern, s)
    params_match = re.search(params_pattern, s)

    url = url_match.group(1) if url_match else None

    headers_str = headers_match.group(1) if headers_match else None
    params_str = params_match.group(1) if params_match else None

    # Convert to valid JSON format
    headers_str = "{" + headers_str + "}" if headers_str else None
    params_str = "{" + params_str + "}" if params_str else None
    
    if headers_str:
        headers_str = headers_str.replace('True', 'true').replace('False', 'false').replace("'", "\"")
    if params_str:
        params_str = params_str.replace('True', 'true').replace('False', 'false').replace("'", "\"")

    # # Replace problematic terms to be JSON-compatible
    # if headers_str:
    #     headers_str = headers_str.replace('true', 'True').replace('false', 'False').replace("'", "\"")
    # if params_str:
    #     params_str = params_str.replace('true', 'True').replace('false', 'False').replace("'", "\"")

    print(f'URL: {url}')
    print(f'Headers: {headers_str}')
    print(f'Params: {params_str}')

    # Convert strings to dictionaries using json.loads
    headers = json.loads(headers_str) if headers_str else None
    params = json.loads(params_str) if params_str else None

    return url, headers, params

def fix_rapidapi_arguments(input_filepath):
    new_data = []
    with open(input_filepath, 'r') as file:
        data = json.load(file)
        
    # pattern = re.compile(r'requests\.get\(\s*"(.*?)",\s*headers=(\{.*?\})(?:,\s*params=(\{.*?\}))?\s*\)', re.DOTALL)
    for object in tqdm(data):
        try:
            model_answer = object["model_answer"]
            url, headers, params = extract_arguments(model_answer)
            # print(f"URL: {url}")
            # print(f"Headers: {headers}")
            # print(f"Params: {params}")
            
            # # pattern = re.compile(r'requests\.get\("(.*?)", headers=(\{.*?\}), params=(\{.*?\})\)')
            # pattern = re.compile(r'requests\.get\("(.*?)", headers=(\{.*?\}), params=(\{.*?\})\)')

            # # Extract the relevant sections of the string
            # match = pattern.search(model_answer)
            # if match is None:
            #     print(f"No match found in model answer: {model_answer}")
            #     continue
            # url, headers, params = match.groups()
            
            if headers is None:
                continue
            
            api_arguments = [
                {"name": "url", "type": "string", "description": "The endpoint URL to which the API request is made. It specifies the location of the resource on the server.", "enum": [url]},
                {"name": "headers", "type": "Dict", "description": "Contains metadata sent with the API request. Headers can include authentication tokens, client information, and other key-value pairs to provide context or directives for the request.", "enum": [headers]},
                {"name": "params", "type": "Dict", "description": "Parameters passed with the API request, typically used to filter or customize the response. They are included in the URL after a question mark (?).", "enum": [params]}
            ]
            object['original']['api_arguments'] = api_arguments

        except Exception as e:
            print(f"Error: {e}")
            print(f"Model Answer: {model_answer}")
            continue
    new_input_filepath = input_filepath.replace(".json", "_fixedArguments.json")
    with open(f'{new_input_filepath}', 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)

def fix_lastchar(input_filepath):
    new_data = []
    with open(input_filepath, 'r') as file:
        data = json.load(file)
    for object in data:
        model_answer = object["model_answer"]
        
        pattern = re.compile(r'(\w+)=["\']?([^"\']+)["\']?')
        # Finding all matches
        matches = pattern.findall(model_answer)

        # Extracting to dictionary
        parameters = {key: value for key, value in matches}

        if not parameters:
            continue
        
        
        skip = False
        for api_argument in object['original']['api_arguments']:
            found = False
            for api_argument_original_key in object['original']['api_arguments_original'].keys():
                if api_argument['name'] in api_argument_original_key:
                    found = True
                    break
            if found == False:
                skip = True
                break
        
        if skip:
            continue
        
        for api_argument in object['original']['api_arguments']:
            if api_argument.get('value', None) is not None:
                del api_argument['value']
        
        skip = False
        for api_argument in object['original']['api_arguments']:
            found = False
            for parameter_key in parameters.keys():
                if api_argument['name'].replace('-', '_') in parameter_key:
                    api_argument['enum'] = [parameters[parameter_key].replace(')', '')]
                    found = True
                    break
            if found == False:
                skip = True
                break
            
        # if skip:
        #     continue
            
        
        new_data.append(object)
        
    new_input_filepath = input_filepath
    with open(f'{new_input_filepath}', 'w') as jsonfile:
        json.dump(new_data, jsonfile, indent=4)

def fix_additional(input_filepath, rapidAPI=False):
    new_data = []
    with open(input_filepath, 'r') as file:
        data = json.load(file)
    for object in data:
        if len(object["model_answer"]) == 0:
            continue
        model_answer = object["model_answer"]
        arguemnt_index = model_answer.find("<Arguments>")
        object['model_answer'] = model_answer[:arguemnt_index].strip()
        argument = model_answer[arguemnt_index:].replace("<Arguments>", "").strip()
        
        arguments = argument.split(";")
        arguments = [{"name":a.split(':')[0].strip(), "enum":[a.split(':')[1].strip()]} for a in arguments if ':' in a]
        for argument in arguments:
            if rapidAPI is True:
                for original_argument in object['original']['api_arguments_original']:
                    if argument['name'] in original_argument['name']:
                        argument['description'] = original_argument['description'].strip()
                        break
            else:
                for original_argument_key in object['original']['api_arguments_original'].keys():
                    if argument['name'] in original_argument_key:
                        argument['description'] = object['original']['api_arguments_original'][original_argument_key]
                        break
        object['original']['api_arguments'] = arguments
        new_data.append(object)
    new_input_filepath = input_filepath.replace(".json", "_fixed.json")
    with open(f'{new_input_filepath}', 'w') as jsonfile:
        json.dump(new_data, jsonfile, indent=4)

def clean_argument_name(argument):
    return argument.strip().replace("--", "").replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace(">", "").replace("<", "").split('=')[0].lower().strip()
    


def fix_arguments(object_json):
    if object_json['original'].get('api_arguments_original', False) is False:
        object_json['original']['api_arguments_original'] = object_json['original']['api_arguments']
    original_arguments = object_json['original']['api_arguments_original']
    
    if isinstance(original_arguments, dict):
        new_arguments = []
        for key in original_arguments.keys():
            new_arguments.append({
                "name": clean_argument_name(key),
                "description": original_arguments[key].strip()
            })
        return new_arguments
    
    if isinstance(original_arguments, list):
        new_arguments = []
        for argument in original_arguments:
            if isinstance(argument, str):
                new_arguments.append({
                    "name": clean_argument_name(argument),
                })
            elif isinstance(argument, dict):
                new_dict = {}
                new_dict['name'] = clean_argument_name(argument['name'])
                if argument.get('description', None) is not None:
                    new_dict['description'] = argument['description'].strip()
                if argument.get('type', None) is not None:
                    new_dict['type'] = argument['type']
                new_arguments.append(new_dict)
        return new_arguments
    
    return original_arguments
    
    if len(object_json['original']['api_arguments']) == 0:
        return original_arguments
    if object_json['original']['api_arguments'][0].get('name', None) is not None:
        return object_json['original']['api_arguments']
    new_arguments = []
    for argument in original_arguments:
        new_arguments.append({
            "name": argument['name'],
            "enum": [argument['value']],
            "description": argument['description']
        })


def fix_api_name(object_json):
    api_name = object_json['original']['api_name']
    return api_name.replace("-", "_").strip()

def fix_value_to_enum(input_filepath):
    with open(input_filepath, 'r') as file:
        data = json.load(file)
    
    # Issue 6: Loop directly over elements.
    for example in tqdm(data, desc='Processing data'):  
        for argument in example['original']['api_arguments']:
            try:
                # Issue 2: Fix incorrect indexing.
                argument["enum"] = argument["value"]
                # Remove the old key
                del argument['value']
                
                # Check if 'enum' is not a list and convert it to a single-item list if needed.
                if not isinstance(argument["enum"], list):
                    argument["enum"] = [argument["enum"]]
                    
            # Issue 5: Handle possible exceptions (customize as needed).
            except KeyError as e:
                print(f"KeyError: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
    
    with open(f'{input_filepath}', 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)
    

def create_additional_examples(input_filepath, max_per_file=None):
    curr_file = input_filepath.split("/")[-1]
    print(f"{curr_file}")
    with open(input_filepath, 'r') as file:
        data = json.load(file)
    
    new_data = []
    
    for index, example in tqdm(enumerate(data), desc='Processing data'):
        if max_per_file is not None and index >= max_per_file:
            break
        tries = 0
        while True:
            if tries >= 3:
                break
            try:
                prompt = template.create_additional_queries.replace("<<<DICT>>>", json.dumps(example, indent=4)).replace("<<<EXAMPLES>>>", prompt_examples.SYNTHETIC_REQUEST_GENERATION)
                response = OpenAI_API.chatgpt(prompt)
                # print(f"\n\nResponse:\n{response}\n\n")
                
                model_answer_inddex = response.find("<New Model Answer>")
                if model_answer_inddex == -1:    
                    print(f"Model Answer not found!")
                    print(response)
                    tries += 1
                    continue
                # arguemnt_index = response.find("<Arguments>")
                
                new_query_index = response.find("<New Query>")
                if new_query_index >= 0:
                    new_query = response[new_query_index:].replace("<New Query>", "").strip()
                else:
                    new_query = response.split("\n")[0][:model_answer_inddex].strip()
                model_answer = response[model_answer_inddex:].replace("<New Model Answer>", "").strip()
                
                ast.parse(model_answer)
                if len(model_answer.strip()) == 0:
                    print("Model Answer is empty!")
                    print(response)
                    tries += 1
                    continue
                
                # argument = response[arguemnt_index:].replace("<Arguments>", "").strip()
                # arguments = argument.split(";")
                # arguments = [{a.split(':')[0].strip():a.split(':')[1].strip()} for a in arguments if ':' in a]
                
                # if example['original'].get('api_arguments_original', False) is False:
                #     example['original']['api_arguments_original'] = example['original']['api_arguments']
                
                arguments = fix_arguments(example)
                
                example['query'] = new_query
                example['model_answer'] = model_answer
                example['original']['api_arguments'] = arguments
                
                example['original']['api_name'] = fix_api_name(example)
                
                new_data.append(example)
                break
            except Exception as e:
                print(f"{index}-{tries}. {e}\n{response}")
                tries += 1
                if tries >= 3:
                    break
                continue
    
    dirs = input_filepath.split("/")
    dirs.insert(-1, "additional")
    new_input_filepath = "/".join(dirs)
    # new_input_filepath = input_filepath.replace(".json", "_additional.json")
    with open(f'{new_input_filepath}', 'w') as jsonfile:
        json.dump(new_data, jsonfile, indent=4)
        
        
    

def get_fixed_python_response(model_answer):
    prompt = template.fix_response_to_python.replace("<<<EXAMPLE_API_CALL>>>", model_answer).replace("<<<EXAMPLES>>>", prompt_examples.FIX_RESPONSE_TO_PYTHON)
            
    response = OpenAI_API.chatgpt(prompt).strip()
    return response

def get_fixed_response(model_answer, index):
    i = 0
    while not check.is_parsable(model_answer):
        if i >= 1:
            print(f"{index}-{i+1}. Model Answer Not Parsable: {model_answer}")
        model_answer = get_fixed_python_response(model_answer)
        i += 1
        if i > 2:
            return False, model_answer
    return True, model_answer
            

def fix_python_parsable(data, clean_input_name):
    new_data = []
    skipped_data = []
    for index, json_object in enumerate(tqdm(data)):
        json_object['model_answer'] = json_object['model_answer'].replace("-", "_")
        no_error, json_object['model_answer'] = get_fixed_response(json_object['model_answer'], index)
        
        if no_error is False:
            skipped_data.append(json_object)
        else:
            new_data.append(json_object)
    
    print(f"Number of skipped examples: {len(skipped_data)}/{len(data)}")
    now = datetime.now()
    # Convert to string format
    date_string = now.strftime('%m_%d_%H_%M')
    
    with open(f'output/{clean_input_name}_{date_string}.json', 'w') as jsonfile:
        json.dump(new_data, jsonfile, indent=4)
        
    with open(f'output/{clean_input_name}_{date_string}_skipped.json', 'w') as jsonfile:
        json.dump(skipped_data, jsonfile, indent=4)

def fix_aws(data, clean_input_name, max=-1):
    output_data = []
    
    errors_in_a_row = 0
    errors_dicts = {}
    
    for index, example in enumerate(tqdm(data)):
        # print(f"Example {index+1}/{len(data)}")
        
        if max != -1:
            if len(output_data) >= max:
                break
        
        try:
            original_json = example['original']
            
            name = original_json['api_name']
            arguments = original_json['api_arguments']
            model_answer = example['model_answer']

            original_json['api_name_original'] = name
            original_json['api_arguments_original'] = arguments
            
            argument_descriptions = []
            
            api_name = None
            
            prompt = template.fix_template_2.replace("<<<EXAMPLE_API_CALL>>>", model_answer).replace("<<<EXAMPLES>>>", prompt_examples.AWS_FIX_EXAMPLES)
            
            response = OpenAI_API.chatgpt(prompt)
            
            responses = [r for r in response.split("\n") if len(r) > 0]
            
            if len(responses) == 0:
                continue
            
            api_arguments = []
            
            for response_index, response in enumerate(responses):
                if response_index == 0:
                    api_name = response
                else:
                    if ";" not in response:
                        print(f"; not in Response: {index+1}/{len(data)}")
                        print(f"Response: {response}")
                        continue
                    argument_name = response.split(";")[0].strip()
                    argument_value = response.split(";")[1].strip()
                    argument_description = ""
                    
                    for key, value in original_json['api_arguments_original'].items():
                        if argument_name in key and len(key) - len(argument_name) <= 3:
                            argument_description = value
                            
                    if len(argument_description) == 0:
                        print(f"Argument Description Not Found: {index+1}/{len(data)}")
                        print(f"Argument Name: {argument_name}")
                        api_argument_original = original_json['api_arguments_original']
                        print(f"Original Arguments: {api_argument_original}")
                        continue
                    
                    api_arguments.append({
                        "name": argument_name,
                        "value": argument_value,
                        "description": argument_description
                    })
                            
            original_json['api_name'] = api_name
            original_json['api_arguments'] = api_arguments
            
            example['original'] = original_json
            output_data.append(example)
            
        except Exception as e:
            print(f"Error: {index+1}/{len(data)}")
            print(e)
            if e in errors_dicts:
                errors_dicts[e] += 1
            else:
                errors_dicts[e] = 1
            
            print(f"\n<Response>\n{response}\n")
            errors_in_a_row += 1
            print(f"Errors: {errors_in_a_row}")
            print(f"{errors_dicts}")
            continue
        
    now = datetime.now()
    # Convert to string format
    date_string = now.strftime('%m_%d_%H_%M')
    
    with open(f'output/{clean_input_name}_{date_string}_cleaned.json', 'w') as jsonfile:
        json.dump(output_data, jsonfile, indent=4)

def fix_file(data, clean_input_name):
    output_data = []
    
    errors_in_a_row = 0
    errors_dicts = {}
    
    for index, example in enumerate(tqdm(data)):
        # print(f"Example {index+1}/{len(data)}")
        
        try:
            # example_str = json.dumps(example)
            
            # domain = example['domain']
            # framework = example['framework']
            # functionality = example['functionality']
            
            original_json = example['original']
            
            name = original_json['api_name']
            arguments = original_json['api_arguments']
            model_answer = example['model_answer']
            
            original_json['api_name'] = 'requests.get'
            original_json['api_name_original'] = name
            original_json['api_arguments_original'] = arguments
            
            url_template = {
                "name": "url",
                "type": "string",
                "description": "The endpoint URL to which the API request is made. It specifies the location of the resource on the server.",
                # "default": "Downing Street London"
            }
            
            headers_template = {
                "name": "headers",
                "type": "Dict",
                "description": "Contains metadata sent with the API request. Headers can include authentication tokens, client information, and other key-value pairs to provide context or directives for the request.",
                # "default": "Downing Street London"
            }
            
            params_template = {
                "name": "params",
                "type": "Dict",
                "description": "Parameters passed with the API request, typically used to filter or customize the response. They are included in the URL after a question mark (?).",
                # "default": "Downing Street London"
            }
            
            # # Regular expression patterns
            # url_pattern = r'requests\.get\(\"(.*?)\"'
            # headers_pattern = r'headers=\{(.*?)\}'
            # params_pattern = r'params=\{(.*?)\}'

            # # Extracting URL, headers and params
            # url = re.search(url_pattern, model_answer).group(1)
            # headers_string = re.search(headers_pattern, model_answer).group(1)
            # params_string = re.search(params_pattern, model_answer).group(1)
            
            # headers = string_to_dict(headers_string)
            # params = string_to_dict(params_string)
            
            # prompt = template.fix_template.format(model_answer)
            prompt = template.fix_template.replace("<<<EXAMPLE_APU_CALL>>>", model_answer)
            
            response = OpenAI_API.chatgpt(prompt)
            
            # print(f"\n<Response>\n{response}\n")
            
            lines = response.split("\n")
            if len(lines) < 3:
                print(f"Response Too Short Error: {index+1}/{len(data)}")
                print(f"\n<Response>\n{response}\n")
                continue
            
            # Extract URL
            url = ""
            if "URL" in response:
                # url_section = response[:response.find("Headers")]
                url = response.split('\n')[0].split(":",1)[1].strip()
            else:
                print(f"URL Not Found Error: {index+1}/{len(data)}")
                print(f"<Response>\n{response}\n")
                continue

            # Extract Headers
            headers = {}
            if "Headers" in response:
                headers_section = response[response.find("Headers"):]
                headers_str = headers_section.split('\n')[0].split(":",1)[1].strip()
                if (headers_str[0] != "{" or headers_str[-1] != "}") and "None" not in headers_str:
                    print(f"Headers Not Found Error: {index+1}/{len(data)}")
                    print(f"<Reader>\n{headers_str}\n")
                    continue
                headers = json.loads(headers_str) if headers_str and "None" not in headers_str else {}
            else:
                continue

            # Extract Params
            params = {}
            if "Params" in response:
                params_section = response[response.find("Params"):]
                params_str = params_section.split('\n')[0].split(":",1)[1].strip()
                if (params_str[0] != "{" or params_str[-1] != "}") and "None" not in params_str:
                    print(f"Params Not Found Error: {index+1}/{len(data)}")
                    print(f"<Reader>\n{params_str}\n")
                    continue
                params = json.loads(params_str) if params_str and "None" not in params_str else {}
            else:
                continue

            # # try:
            # url = response.split("\n")[0].split(": ", 1)[1].strip()
            # headers = response.split("\n")[1].split(":", 1)[1].strip()
            # if "None" in headers: headers = {}
            # else:
            #     headers = json.loads(headers)
            # params = response.split("\n")[2].split(":", 1)[1].strip()
            # if "None" in params: params = {}
            # else:
            #     params = json.loads(params)
                
            # print(f"URL: {url}")
            # print(f"Headers: {headers}")
            # print(f"Params: {params}")
            
            url_template['value'] = url
            headers_template['value'] = headers
            params_template['value'] = params
            
            original_json['api_arguments'] = [url_template, headers_template, params_template]
            example['original'] = original_json
            sample_dict = example
            
            output_data.append(sample_dict)
        except Exception as e:
            # print(e)
            print(f"Error: {index+1}/{len(data)}")
            print(e)
            if e in errors_dicts:
                errors_dicts[e] += 1
            else:
                errors_dicts[e] = 1
            
            print(f"\n<Response>\n{response}\n")
            errors_in_a_row += 1
            print(f"Errors in a row: {errors_in_a_row}")
            # if errors_in_a_row > 10:
            #     print("Too many errors in a row. Stopping.")
            #     break
            continue
    now = datetime.now()
    # Convert to string format
    date_string = now.strftime('%m_%d_%H_%M')
    
    with open(f'output/{clean_input_name}_{date_string}_cleaned.json', 'w') as jsonfile:
        json.dump(output_data, jsonfile, indent=4)

def abstract_fix(file_path):
    output_data = []
    
    data = json.load(open(file_path))
    for index, json_obj in enumerate(tqdm(data)):
        try:    
            ast.parse(json_obj['model_answer'])
            
            if json_obj['original'].get('api_arguments_original', False) is False:
                json_obj['original']['api_arguments_original'] = json_obj['original']['api_arguments']
            
            arguments = fix_arguments(json_obj)
            json_obj['original']['api_arguments'] = arguments
            
            json_obj['original']['api_name'] = fix_api_name(json_obj)
            
            output_data.append(json_obj)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    dirs = file_path.split("/")
    dirs.insert(-1, "original")
    new_input_filepath = "/".join(dirs)
    # Create the directory if it doesn't exist
    new_dir_path = "/".join(dirs[:-1])
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    # new_input_filepath = input_filepath.replace(".json", "_additional.json")
    with open(f'{new_input_filepath}', 'w') as jsonfile:
        json.dump(output_data, jsonfile, indent=4)
    

def main():
    # input_file = rapidAPI
    # input_file = aws
    # data = json.load(open(f'input/{input_file}'))
    # clean_input_name = input_file.split(".")[0]
    # fix_file(data, clean_input_name)
    # fix_aws(data, clean_input_name)
    
    # data = json.load(open(f'output/{output_aws}'))
    # clean_input_name = output_aws.split(".")[0]
    # fix_python_parsable(data, clean_input_name)
    
    latest_aws = "output/aws-cli-2023_09_29_gpt_3_5_turbo_10_08_00_00_cleaned_10_12_20_48_additional_fixed_fixed.json"
    # fix_lastchar(latest_aws)
    # fix_additional(latest_aws)
    # fix_value_to_enum(latest_aws)
    
    dataset1 = ['output/dataset1/aws-cli-2023_09_29_gpt_3_5_turbo_10_08_00_00_cleaned_10_12_20_48_additional_fixed_fixed.json',
               'output/dataset1/openai_gcloud-2023Jun13_fixed_10_19_fixed_fixed.json',
               'output/dataset1/openai_github-2023Jun13_fixed_10_19_fixed_fixed.json',
               'output/dataset1/rapidAPI-api_09_30_gpt_3_5_turbo_10_06_18_53_cleaned_additional_fixed_fixedArguments_fixed.json',
               'output/dataset1/rapidAPI-api_09_30_gpt_3_5_turbo_10_06_18_53_cleaned_fixed_fixed.json']
    for dataset in dataset1:
        abstract_fix(dataset)
    
    latset_rapid = "output/rapidAPI-api_09_30_gpt_3_5_turbo_10_06_18_53_cleaned_additional_fixed.json"
    # fix_rapidapi_arguments(latset_rapid)
    # fix_additional(latset_rapid, rapidAPI=True)
    # create_additional_examples(latset_rapid)
    # fix_value_to_enum(latset_rapid)
    
    rapidapi = 'output/rapidAPI-api_09_30_gpt_3_5_turbo_10_06_18_53_cleaned_additional_fixed_fixedArguments.json'
    # fix_rapidapi(rapidapi)
    
    gcloud = "output/openai_gcloud-2023Jun13_fixed_10_19.json"
    github = "output/openai_github-2023Jun13_fixed_10_19.json"
    # fix_gcloud(github)
    
    gcloud = "output/openai_gcloud-2023Jun13_fixed_10_19_fixed.json"
    github = "output/openai_github-2023Jun13_fixed_10_19_fixed.json"
    # fix_model_names(gcloud)
    
    

if __name__ == "__main__":
    main()