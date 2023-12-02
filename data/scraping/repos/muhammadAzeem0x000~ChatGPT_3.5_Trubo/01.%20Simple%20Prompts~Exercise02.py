### FREEZE CODE BEGIN
import openai
import os

openai.api_key = os.getenv('OPENAI_KEY')
### FREEZE CODE END
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "Your are expert assistant in python programming."},# make sure to replace empty string
        {"role": "user", "content":"make a function to reverse the string." }# make sure to replace empty string
    ]
)

print(response['choices'][0]['message']['content'].strip())