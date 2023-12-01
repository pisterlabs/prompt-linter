import os
import openai
import ipdb

if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_KEY")
    resp = openai.File.create(file=open("local_data/training_data/output.jsonl", "rb"), purpose="fine-tune")
    openai.FineTuningJob.create(training_file=resp["id"], model="gpt-3.5-turbo")
    ipdb.set_trace()
