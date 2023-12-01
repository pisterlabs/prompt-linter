import openai
import os

# Set your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set the name of the file you want to upload
filename = "gitpod_qa_pairs.jsonl"

# Upload the training data
print(f"Uploading {filename} as training data...")
with open(filename, "rb") as f:
    training_data_response = openai.File.create(
        purpose="fine-tune",
        file=f
    )

# Print the response from OpenAI
print(f"Uploaded {filename} with file ID: {training_data_response.id}")
