import json
import os
import openai
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config['openai']['api_key']

# Define the validation function
def validate_jsonl_object(json_obj):
    if "prompt" not in json_obj or "completion" not in json_obj:
        return False
    return True

# Initialize row count
row_count = 0

# Get file path from the user
file_path = input("Please enter the path to your JSONL file: ")

# Open the JSONL file for reading
try:
    with open(file_path, "r") as f:
        for line in f:
            row_count += 1
            json_obj = json.loads(line.strip())
            
            # Validate the JSON object
            if not validate_jsonl_object(json_obj):
                print(f"Validation failed for object at row {row_count}: {json_obj}")

    # Print the total row count
    print(f"Validation completed. Total rows processed: {row_count}")

    # If you reached this point, you can proceed to upload the file to OpenAI
    openai.api_key = os.getenv(openai_api_key)
    openai.File.create(
      file=open(file_path, "rb"),
      purpose='fine-tune'
    )
    print("File uploaded to OpenAI for fine-tuning.")

# Handle file not found error
except FileNotFoundError:
    print(f"File at path {file_path} was not found. Please check the path and try again.")
