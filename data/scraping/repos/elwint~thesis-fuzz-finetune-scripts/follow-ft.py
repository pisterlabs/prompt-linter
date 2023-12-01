import openai
import os
import sys

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.FineTuningJob.retrieve(sys.argv[1])

print(response)
