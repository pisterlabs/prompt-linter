"""
Using the python API to prepare dataset and then tune models
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=4, help='Number of epochs to run')
args = parser.parse_args()
n_epochs = args.n_epochs

import os
import openai
# Internal imports
from params import models
from utils import wait_for_messages, find_uploaded_data, find_finetuned_models, set_openai_keys

# Make sure keys are set
set_openai_keys()

###################################
# --- (1) UPLOAD/TRAIN MODELS --- #

di_uploaded_data = find_uploaded_data()
di_finetuned_models = find_finetuned_models()


for model in models:
    print(f"Fine-tuning model: {model}")
    # Specify path to the jsonl file
    path = f"data/training_data_{model}.jsonl"
    # Ensure the file exists
    assert os.path.exists(path), f"File {path} does not exist"

    # Check to see if model file has already been uploaded to OpenAI
    if model in di_uploaded_data:
        print(f"File already uploaded")
    else:
        print(f"Uploading file")
        data_upload = openai.File.create(file=open(path, "rb"), purpose='fine-tune', user_provided_filename=model)
    # Get the id of the uploaded file
    data_id = di_uploaded_data[model]
    
    # Save parameter information for later
    if model in di_finetuned_models:
        print(f"Model already trained")
    else:
        print(f"Training model")
        finetune = openai.FineTune.create(training_file=data_id, model=model, n_epochs=n_epochs, suffix='econchattr')
        # Retrieve the status
        openai_id = finetune['id']
        wait_for_messages(openai_id)
        input(f"Congratulations model {finetune['model']} has finished training, press Enter to continue...")



print('~~~ End of 4_tune_models.py ~~~')