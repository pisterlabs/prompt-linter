import json

training_data = 'train_labeled_5000_balanced.csv'
validation_data = 'validation_labeled_2000_balanced.csv'

training_file_name = "input/training_data.jsonl"
validation_file_name = "input/validation_data.jsonl"

MAX_SIZE = 100


def prepare_data(data_file, final_file_name):
    count = 0
    with open(data_file, "r") as data:
        with open(final_file_name, 'w') as outfile:
            data.readline()  # skip header
            for entry in data:
                # count += 1
                # if count >= MAX_SIZE:
                #     break
                if len(entry) < 3:
                    continue
                classification = 'Sarcastic' if entry[-2] == "1" else 'Not Sarcastic'
                out = {"prompt": entry[:-2], "completion": " " + classification + "\n"}
                json.dump(out, outfile)
                outfile.write('\n')

def prepare():
    prepare_data(training_data, training_file_name)
    prepare_data(validation_data, validation_file_name)

###
# next, we will prepare the files using these commands:
# openai tools fine_tunes.prepare_data -f "training_data.jsonl"
# openai tools fine_tunes.prepare_data -f "validation_data.jsonl"
# remember to save the file ID for the fine tuning call
# https://www.datacamp.com/tutorial/fine-tuning-gpt-3-using-the-open-ai-api-and-python
###


###
# output for the upload:
# Now use that file when fine-tuning:
# > openai api fine_tunes.create -t "training_data_prepared.jsonl"
#
# After you’ve fine-tuned a model, remember that your prompt has to end with the indicator string `,` for the model to start generating completions, rather than continuing with the prompt. Make sure to include `stop=[" Sarcastic\n"]` so that the generated texts ends at the expected place.


import openai

# TODO add api keys:
# openai.organization = ""
# openai.api_key = ""

# upload the files and save the ids

def upload_files():
    train = openai.File.create(
      file=open("training_data_prepared.jsonl", "rb"),
      purpose='fine-tune'
    )

    validate = openai.File.create(
      file=open("input/validation_data.jsonl", "rb"),
      purpose='fine-tune'
    )

    print(train)
    print(validate)


# create a fine tuned model

def create_ft():
    ft = openai.FineTune.create(
        model="ada",
        training_file="file-i0y8DIeD4jy4gGTwPbkOg5Fi",
        validation_file="file-1u85iwwaLopl5Faotl7rJ2yH",
        classification_positive_class=" Sarcastic\n",
        compute_classification_metrics=True
    )

    # created with id 'ft-PEKUja5cThAHzfQB8kXilU8y'

    print(ft)
    # {
    #     "object": "fine-tune",
    #     "id": "ft-FOW9iRpuSmPQzm4hdTPjrqNF",
    #     "hyperparams": {
    #         "n_epochs": 4,
    #         "batch_size": null,
    #         "prompt_loss_weight": 0.01,
    #         "learning_rate_multiplier": null,
    #         "classification_positive_class": " Sarcastic\n",
    #         "compute_classification_metrics": true
    #     },
    #     "organization_id": "org-iK29pb2f4dnDuQLgrmwRD5qm",
    #     "model": "ada",
    #     "training_files": [
    #         {
    #             "object": "file",
    #             "id": "file-i0y8DIeD4jy4gGTwPbkOg5Fi",
    #             "purpose": "fine-tune",
    #             "filename": "file",
    #             "bytes": 773326,
    #             "created_at": 1694470145,
    #             "status": "processed",
    #             "status_details": null
    #         }
    #     ],
    #     "validation_files": [
    #         {
    #             "object": "file",
    #             "id": "file-1u85iwwaLopl5Faotl7rJ2yH",
    #             "purpose": "fine-tune",
    #             "filename": "file",
    #             "bytes": 157545,
    #             "created_at": 1694469975,
    #             "status": "processed",
    #             "status_details": null
    #         }
    #     ],
    #     "result_files": [],
    #     "created_at": 1694536881,
    #     "updated_at": 1694536881,
    #     "status": "pending",
    #     "fine_tuned_model": null,
    #     "events": [
    #         {
    #             "object": "fine-tune-event",
    #             "level": "info",
    #             "message": "Created fine-tune: ft-FOW9iRpuSmPQzm4hdTPjrqNF",
    #             "created_at": 1694536881
    #         }
    #     ]
    # }
    # {
    #     "object": "list",
    #     "data": [],
    #     "has_more": false
    # }
    # {
    #     "object": "fine-tune",
    #     "id": "ft-fifD7uNr3mcXKKNIs1NZAVMX",
    #     "hyperparams": {
    #         "n_epochs": 4,
    #         "batch_size": 8,
    #         "prompt_loss_weight": 0.01,
    #         "learning_rate_multiplier": 0.1,
    #         "classification_positive_class": "Sarcastic",
    #         "compute_classification_metrics": true
    #     },
    #     "organization_id": "org-iK29pb2f4dnDuQLgrmwRD5qm",
    #     "model": "ada",
    #     "training_files": [
    #         {
    #             "object": "file",
    #             "id": "file-i0y8DIeD4jy4gGTwPbkOg5Fi",
    #             "purpose": "fine-tune",
    #             "filename": "file",
    #             "bytes": 773326,
    #             "created_at": 1694470145,
    #             "status": "processed",
    #             "status_details": null
    #         }
    #     ],
    #     "validation_files": [
    #         {
    #             "object": "file",
    #             "id": "file-1u85iwwaLopl5Faotl7rJ2yH",
    #             "purpose": "fine-tune",
    #             "filename": "file",
    #             "bytes": 157545,
    #             "created_at": 1694469975,
    #             "status": "processed",
    #             "status_details": null
    #         }
    #     ],
    #     "result_files": [],
    #     "created_at": 1694510985,
    #     "updated_at": 1694511002,
    #     "status": "failed",
    #     "fine_tuned_model": null,
    #     "events": [
    #         {
    #             "object": "fine-tune-event",
    #             "level": "info",
    #             "message": "Created fine-tune: ft-fifD7uNr3mcXKKNIs1NZAVMX",
    #             "created_at": 1694510985
    #         },
    #         {
    #             "object": "fine-tune-event",
    #             "level": "info",
    #             "message": "Fine-tune costs $0.32",
    #             "created_at": 1694510999
    #         },
    #         {
    #             "object": "fine-tune-event",
    #             "level": "info",
    #             "message": "Fine-tune enqueued. Queue number: 0",
    #             "created_at": 1694511000
    #         },
    #         {
    #             "object": "fine-tune-event",
    #             "level": "info",
    #             "message": "Fine-tune started",
    #             "created_at": 1694511001
    #         },
    #         {
    #             "object": "fine-tune-event",
    #             "level": "error",
    #             "message": "The positive class specified `Sarcastic` is not found among the classes in the validation file [' Not Sarcastic\\n', ' Sarcastic\\n']. Fine-tune failed. For help, please contact OpenAI and include your fine-tune ID: ft-fifD7uNr3mcXKKNIs1NZAVMX",
    #             "created_at": 1694511002
    #         }
    #     ]
    # }

# see the model status

def print_status():
    status = openai.FineTune.retrieve("ft-FOW9iRpuSmPQzm4hdTPjrqNF")
    print(status)
#     {
#       "object": "fine-tune-event",
#       "level": "info",
#       "message": "Completed epoch 3/4",
#       "created_at": 1694538278
#     },
#     {
#       "object": "fine-tune-event",
#       "level": "info",
#       "message": "Uploaded model: ada:ft-tau-2023-09-12-17-09-36",
#       "created_at": 1694538576
#     },
#     {
#       "object": "fine-tune-event",
#       "level": "info",
#       "message": "Uploaded result file: file-AuIoJJSOKY5jXpH4iIruuPoe",
#       "created_at": 1694538577
#     },
#     {
#       "object": "fine-tune-event",
#       "level": "info",
#       "message": "Fine-tune succeeded",
#       "created_at": 1694538577
#     }
#   ]
# }

def print_results():
    content = openai.File.download("file-AuIoJJSOKY5jXpH4iIruuPoe")
    contents = content.decode()
    print(contents)