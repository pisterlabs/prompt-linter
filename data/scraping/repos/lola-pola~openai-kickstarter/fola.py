import openai
from openai import cli
import time
import shutil
import json

# Remember to remove your key from your code when you're done.
openai.api_key = "COPY_YOUR_OPENAI_KEY_HERE"
# Your resource endpoint should look like the following:
# https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_base =  "COPY_YOUR_OPENAI_ENDPOINT_HERE" 
openai.api_type = 'azure'
# The API version may change in the future.
openai.api_version = '2023-05-15'

training_file_name = 'training.jsonl'
validation_file_name = 'validation.jsonl'

sample_data = [{"prompt": "When I go to the store, I want an", "completion": "apple"},
    {"prompt": "When I go to work, I want a", "completion": "coffee"},
    {"prompt": "When I go home, I want a", "completion": "soda"}]

# Generate the training dataset file.
print(f'Generating the training file: {training_file_name}')
with open(training_file_name, 'w') as training_file:
    for entry in sample_data:
        json.dump(entry, training_file)
        training_file.write('\n')

# Copy the validation dataset file from the training dataset file.
# Typically, your training data and validation data should be mutually exclusive.
# For the purposes of this example, we're using the same data.
print(f'Copying the training file to the validation file')
shutil.copy(training_file_name, validation_file_name)

def check_status(training_id, validation_id):
    train_status = openai.File.retrieve(training_id)["status"]
    valid_status = openai.File.retrieve(validation_id)["status"]
    print(f'Status (training_file | validation_file): {train_status} | {valid_status}')
    return (train_status, valid_status)

# Upload the training and validation dataset files to Azure OpenAI.
training_id = cli.FineTune._get_or_upload(training_file_name, True)
validation_id = cli.FineTune._get_or_upload(validation_file_name, True)

# Check on the upload status of the training and validation dataset files.
(train_status, valid_status) = check_status(training_id, validation_id)

# Poll and display the upload status once a second until both files have either
# succeeded or failed to upload.
while train_status not in ["succeeded", "failed"] or valid_status not in ["succeeded", "failed"]:
    time.sleep(1)
    (train_status, valid_status) = check_status(training_id, validation_id)
