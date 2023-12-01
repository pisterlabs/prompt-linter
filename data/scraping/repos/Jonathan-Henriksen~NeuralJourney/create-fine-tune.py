import os
import time
import openai
import argparse

# Parsing command line arguments
parser = argparse.ArgumentParser(description="OpenAI Fine-Tuning Script")
parser.add_argument('--file_path', required=True, help='Path to the file for training')
parser.add_argument('--file_name', required=True, help='Name of the training file')
parser.add_argument('--model_name', required=True, help='Name of the model to use')
args = parser.parse_args()

file_path = args.file_path
file_name = args.file_name
model_name = args.model_name

# Creating the file
file = openai.File.create(
  file=open(file_path, "rb"),
  purpose='fine-tune',
  user_provided_filename=file_name
)

print(file)

# Waiting for the file to be processed
while True:
    file_status = openai.File.retrieve(file.id).status
    if file_status == "processed":
        break
    print(f"File status: {file_status}. Waiting for file to be processed..")
    time.sleep(1)

# Setting API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Creating the fine-tuning job
fineTuneJob = openai.FineTuningJob.create(training_file=file.id, model=model_name)

print(fineTuneJob)