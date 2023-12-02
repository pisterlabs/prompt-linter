import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.organization = "org-D2FBgBhwLFkKAOsgtSp86b4i"
openai.api_key = os.getenv("OPENAI_API_KEY")

remote_files = openai.File.list()["data"]
fine_tune_training_files = filter(lambda f: "fine_tune.jsonl" in f["filename"], remote_files)
latest_fine_tune_training_file = max(fine_tune_training_files, key=lambda x: x["created_at"])
fine_tune_validation_files = filter(lambda f: "fine_tune.jsonl" in f["filename"], remote_files)
latest_fine_tune_validation_file = max(fine_tune_validation_files, key=lambda x: x["created_at"])

fine_tunes = openai.FineTune.list()["data"]
latest_fine_tune_model = max(fine_tunes, key=lambda x: x["created_at"])

print(f"Latest fine-tuned model: {latest_fine_tune_model['id']}")
print(f"Latest fine-tuned model created at: {latest_fine_tune_model['created_at']}")
print(f"Latest fine-tune training file id: {latest_fine_tune_training_file['id']}")
print(f"Latest fine-tune training file created at: {latest_fine_tune_training_file['created_at']}")
print(f"Latest fine-tune validation file id: {latest_fine_tune_validation_file['id']}")
print(f"Latest fine-tune validation file created at: {latest_fine_tune_validation_file['created_at']}")