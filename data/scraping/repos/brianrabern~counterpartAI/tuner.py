import openai
import os
from dotenv import load_dotenv
import subprocess

# need OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# path to training data
training_data_file = "training_data_prepared.jsonl"
model = "currie"  # or davinci or ada, etc.

# define the fine-tuning command
fine_tune = [
    "openai",
    "api",
    "fine_tunes.create",
    "-t",
    training_data_file,
    "-m",
    model,
    "-e",
    api_key,
]

# tune it
subprocess.run(fine_tune, check=True)
