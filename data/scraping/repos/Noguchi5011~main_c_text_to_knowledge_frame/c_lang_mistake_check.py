import openai
import os
import json
from natsort import natsorted
from dotenv import dotenv_values

def read_file(filename):
    lines = []
    with open(filename, 'r') as file:
        lines = file.readlines()
    return lines

def count_lines(lines):
    count = len(lines)
    return count

def get_sorted_c_files(folder_path):
    c_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".c"):
            c_files.append(file_name)
    sorted_c_files = natsorted(c_files)
    return sorted_c_files

def get_correct_program(mistake_bool, explain_text, error_text, correct_program):
    return {
        "mistake_bool": mistake_bool,
        "explain_text": explain_text,
		"error_text": error_text,
        "correct_program": correct_program
    }

def run_conversation(program):
    messages = [{"role": "user", "content": f"{program} From the above program, generate the error, the correct program, and the operation of the correct program. Also, please generate the operation details of the correct program in Japanese."}]
    
    functions = [{
        "name": "get_correct_program",
        "description": "Obtain text in Japanese from the input program with or without errors, the correct program, and the program's operation details.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mistake_bool":{
                        "type":"string",
                        "description":"Whether there is a mistake in the given code or not return a Boole value." ,
                        "enum": ["True", "False"]
                        },
                        "explain_text":{
                        "type":"string",
                        "description":"Explain the operation of the correct program in Japanese."   
                        },
                        "error_text":{
                        "type":"string",
                        "description":"Explain the error of the program in Japanese."   
                        },
                        "correct_program":{
                        "type":"string",
                        "description":"Correct code."
                        }
                    },
                        "required": ["mistake_bool","explain_text","error_text","correct_program"]
                }
            }

        ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    response_data = {
        "id": response["id"],
        "object": "chat.completion",
        "created": response["created"],
        "model": "gpt-3.5-turbo-0613",
        "choices": response["choices"],
        "usage": response["usage"]
    }
    return response_data, ''.join(program)  # Assuming program is a list of lines

config = dotenv_values("../.env")

openai.api_key = config["api_key"]

program_folder = '../input/'
file_list = get_sorted_c_files(program_folder)

single_file = file_list[0]
program_content = read_file(os.path.join(program_folder, single_file))

response_data, program_content_string = run_conversation(program_content)

output_format = [
    response_data,
    program_content_string
]

print(output_format)

with open(f'./generate/correct_combined_results.json', 'w') as result_file:
    json.dump(output_format, result_file, ensure_ascii=False, indent=4)
