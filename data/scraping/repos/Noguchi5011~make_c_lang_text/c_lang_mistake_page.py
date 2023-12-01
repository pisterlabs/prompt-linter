import openai
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import dotenv_values
import json
from natsort import natsorted

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

json_template = {
    "line_number": "〇",
    "mistake_bool": "〇行目の間違いの有無",
    "mistake": {
        "missing_character": "〇行目で抜けている文字",
        "start_position": "開始位置",
        "end_position": "終了位置"
    },
    "question": "〇行目のプログラムの間違いについて聞き出す質問文",
    "correct_program": "正解のプログラム"
}

config = dotenv_values(".env")

openai.api_key = config["api_key"]

program_folder = 'input/'
file_list = get_sorted_c_files(program_folder)

results = {}  # Add this line

for filename in file_list:
    program_file = filename
    prompt_directory = 'prompt'
    
    program_file_pass = os.path.join(program_folder, program_file)
    with open(program_file_pass, 'r') as file:
        program_data = file.read()

    lines = read_file(program_file_pass)
    line_count = count_lines(lines) 

    for i in range(line_count):
        if lines[i].strip() == '':
            continue        
        template = json.dumps(json_template, ensure_ascii=False, indent=4).replace('"〇"', f'"{i+1}"')
        prompt = f"{program_data}\n上記のC言語プログラムがある。{i+1}行目について\n{template}\nこのjson形式のテンプレートに合わせて結果を出力してください。"
        prompt_file = f"{filename.replace('.c','')}_{i+1}行目.txt"
        prompt_file_name = f"{filename.replace('.c','')}_{i+1}行目"
        with open(os.path.join(prompt_directory, prompt_file), 'w') as prompt_files:
            prompt_files.write(prompt)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.6 + 0.4 / 9
        )

        # Save the result to the Python dictionary
        results[f"line_{i+1}"] = response['choices'][0]['message']['content'].strip()

        time.sleep(30)

# Save the entire result to a JSON file
with open('combined_results.json', 'w') as result_file:
    json.dump(results, result_file, ensure_ascii=False, indent=4)