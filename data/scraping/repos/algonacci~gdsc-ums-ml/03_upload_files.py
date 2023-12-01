from openai import OpenAI

client = OpenAI(
    api_key=""
)
client.files.create(
    file=open("train.jsonl", "rb"),
    purpose="fine-tune"
)
