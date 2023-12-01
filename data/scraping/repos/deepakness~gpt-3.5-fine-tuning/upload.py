import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

def fine_tune_model(file_path):
    # Upload the file
    with open(file_path, "rb") as file_data:
        upload_response = openai.File.create(
        file=file_data,
        purpose='fine-tune'
    )
    
    file_id = upload_response.id
    print(f"File uploaded successfully. ID: {file_id}")

# Usage
fine_tune_model("data.jsonl")