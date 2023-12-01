import openai

openai.api_key = "YOUR API KEYS"

with open("data_prepared.jsonl", "rb") as dataset_file:
    response = openai.Dataset.create(
        file=dataset_file,
        purpose="fine-tuning",
        name="questions_answers_dataset"
    )

dataset_id = response["id"]
