from openai import OpenAI

api_key = f""

client = OpenAI(api_key=api_key)

print(client.fine_tuning.jobs.list(limit=10))
