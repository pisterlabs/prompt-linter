import openai
import requests
from datetime import datetime
import os

# Get your OpenAI API key from environment variable
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Function to download a file from OpenAI API
def download_file(file_url, destination_path):
    response = requests.get(file_url)
    with open(destination_path, 'wb') as file:
        file.write(response.content)

# Function to upload a file to OpenAI API
def upload_file(file_path, file_name):
    with open(file_path, 'rb') as file:
        files = {'file': (file_name, file)}
        response = openai.File.create(file=files)
        return response['id']

# Function to modify a comment line with a timestamp
def modify_file(file_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):
            lines[i] = f'# Modified at {timestamp}\n'
            break
    with open(file_path, 'w') as file:
        file.writelines(lines)

# Replace with your file URL and desired file name
file_url = 'YOUR_FILE_URL'
file_name = 'your_file.md'

# Download the file
download_file(file_url, file_name)

# Modify the file
modify_file(file_name)

# Upload the modified file
file_id = upload_file(file_name, file_name)

def create_jsonl_from_text_file(input_file_path, output_jsonl_path):
    # Leggi il contenuto del file di testo
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()

    # Crea un file JSONL con ogni riga come oggetto JSON
    with open(output_jsonl_path, 'w', encoding='utf-8') as output_file:
        for line in lines:
            json_line = {"text": line.strip()}
            output_file.write(json.dumps(json_line, ensure_ascii=False) + '\n')

create_jsonl_from_text_file("acc.py", "acc.jsonl")

print(f"File '{file_name}' modified and uploaded successfully. File ID: {file_id}")
