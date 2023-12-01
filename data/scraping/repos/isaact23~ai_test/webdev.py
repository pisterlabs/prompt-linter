import openai

TEMPERATURE = 0.6

messages = [
  {"role": "system", "content": "Generate a website based on the user prompt."},
  {"role": "user", "content": "Create a commerce website."}
]

result = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-0613",
  messages=messages,
  temperature=TEMPERATURE
)

print(result)