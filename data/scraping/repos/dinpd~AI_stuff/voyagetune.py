import os
import openai
import time

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key: {openai.api_key}")

# Check for existing active jobs
active_jobs = openai.FineTuningJob.list(status="running")
if active_jobs["data"]:
    print("An active job already exists. Exiting.")
    exit()

# Upload the training data file
file_upload = openai.File.create(
    file=open("voyageai.jsonl", "rb"),
    purpose='fine-tune'
)
file_id = file_upload["id"]
print(f"File ID: {file_id}")

# Incremental backoff
initial_delay = 30  # start with a 30-second delay
max_delay = 3000  # maximum 50 minutes
current_delay = initial_delay

# Loop to repeatedly check file status
while True:
    print(f"Waiting for {current_delay} seconds for the file to be processed...")
    time.sleep(current_delay)
    try:
        job = openai.FineTuningJob.create(
            training_file=file_id,
            model="gpt-3.5-turbo"
        )
        job_id = job["id"]
        print(f"Fine-Tuning Job ID: {job_id}")
        break
    except openai.error.OpenAIError as e:
        print(f"An error occurred: {e}")
        if current_delay < max_delay:
            current_delay *= 2  # double the delay time for the next round
            current_delay = min(current_delay, max_delay)
        else:
            print("Max delay reached. Exiting.")
            exit()

# Monitor the job until it's done or fails
while True:
    job_status = openai.FineTuningJob.retrieve(job_id)
    status = job_status["status"]
    print(f"Job status: {status}")

    if status in ["succeeded", "failed"]:
        break

    time.sleep(60)
