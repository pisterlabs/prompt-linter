import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

c = openai.File.create(
  file=open("data.jsonl", "rb"),
  purpose='fine-tune'
)

print(c)

r = openai.FineTuningJob.create(training_file=c, model="gpt-3.5-turbo", suffix="test-baustore-prod")

print(r)