import openai
import os
import argparse


parser = argparse.ArgumentParser(description='Generate a prompt for a given paper and table.')
parser.add_argument('--prompt_path', type=str, help='Path to the prompt file.')
parser.add_argument('--example_paper_path', type=str, help='Path to the example paper file.')
parser.add_argument('--example_table_path', type=str, help='Path to the example table.')
parser.add_argument('--table_path', type=str, help='Path to the table file.')
parser.add_argument('--paper_path', type=str, help='Path to the paper file to be curated')
args = parser.parse_args()



def generate_prompt(prompt_file, paper_file, table_file, ex_paper,ex_table):
    prompt = load_text_file(prompt_file)
    paper = load_text_file(paper_file)
    table = load_text_file(table_file)
    example_paper = load_text_file(ex_paper)
    example_table= load_text_file(ex_table)
    
    prompt = prompt.replace("[PAPER SPLIT1]", example_paper)
    prompt = prompt.replace("[TABLE SPLIT1]", example_table)
    prompt = prompt.replace("[PAPER SPLIT]", paper)
    prompt = prompt.replace("[TABLE SPLIT]", table)

    return prompt

def load_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

file_path = 'prompt.txt'
prompt = generate_prompt(args.prompt_path, args.paper_path, args.table_path,args.example_paper_path, args.example_table_path )
openai.api_key = "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"
response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You extract information from documents and return json objects"},{"role": "user", "content": prompt}], temperature=0.0)
print(response["choices"][0]["message"]["content"])