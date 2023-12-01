import openai

openai.api_key = "OPENAIKEY"
openai.api_base = "http://localhost:8000/v1"

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "i hate you !!"}
  ],
  headers={
        "whylabs_dataset_id": "model-15"
  }
)

print(response)