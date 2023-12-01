import openai
import os

# Replace with your OpenAI API key
openai.api_key = "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"

input_file = "gitpod_qa_pairs.jsonl"

# Upload the training data
with open(input_file, "rb") as f:
    response = openai.Dataset.create(
        file=f,
        purpose="fine-tuning",
    )

dataset_id = response["id"]
print(f"Uploaded {input_file} to OpenAI with dataset ID: {dataset_id}")

# then run python upload_training_data.py
