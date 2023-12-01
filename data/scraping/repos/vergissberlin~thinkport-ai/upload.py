from openai import OpenAI
client = OpenAI()

client.files.create(
  file=open("./data/training.jsonl", "rb"),
  purpose="fine-tune"
)
