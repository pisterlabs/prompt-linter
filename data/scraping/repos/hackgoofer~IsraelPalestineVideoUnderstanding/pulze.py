import openai
openai.api_key = "sk-UsWepF7VXifvT9M5S4GfVDyo3Kpltc1kC80ondi8P-45M1gpLQff7JbZro89bqUf" # supply your key however you choose
openai.api_base = "https://api.pulze.ai/v1" # enter Pulze's URL

text_response = openai.Completion.create(
  model="pulze-v0",
  prompt="Say Hello World!",
)

chat_response = openai.ChatCompletion.create(
  model="pulze-v0",
  messages=[{
    "role": "user",
    "content": "Say Hello World!"
  }],
)

print(text_response, chat_response)