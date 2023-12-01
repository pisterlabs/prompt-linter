# setup connection to openai api
import openai

openai.api_key = "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"

# create a prompt
prompt = """Bonjour"""
# call openai api using the model davinci

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": prompt},
  ]
)

# print the response

print(response.choices[0].message.content)

print(response)