import time
import openai
from openai import cli

training_file_name = './data/training_data.jsonl'
validation_file_name = './data/validation_data.jsonl'

def check_status(training_file_id, validation_file_id):
    train_status = openai.File.retrieve(training_file_id)["status"]
    valid_status = openai.File.retrieve(validation_file_id)["status"]
    print(f'Status (training_file | validation_file): {train_status} | {valid_status}')
    return (train_status, valid_status)

#importing our two files
training_file_id = cli.FineTune._get_or_upload(training_file_name, True)
validation_file_id = cli.FineTune._get_or_upload(validation_file_name, True)

#checking the status of the imports
(train_status, valid_status) = check_status(training_file_id, validation_file_id)

while train_status not in ["succeeded", "failed"] or valid_status not in ["succeeded", "failed"]:
    time.sleep(1)
    (train_status, valid_status) = check_status(training_file_id, validation_file_id)
