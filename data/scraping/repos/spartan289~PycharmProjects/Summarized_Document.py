import os
import openai
openai.api_key = "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"

user_input=''

with open('File.txt','r') as file:
    user_input=file.read()

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a text summarizer chat bot. your goal is to summarize the text that is provided by the user"},
    {"role": "user", "content": user_input}
  ],
    temperature=0

)
ai_response = response.choices[0].message.content
print("Summarized text: \n\n,",ai_response)

with open('summary.txt','w') as file:
    file.write(ai_response)

