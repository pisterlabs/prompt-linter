from openai import OpenAI

client = OpenAI(
    api_key=""
)

client.fine_tuning.jobs.create(
    training_file="file-E2Da2VERheeT64AlRAQaNq2V",
    model="gpt-3.5-turbo"
)
