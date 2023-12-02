import os
import openai

openai.api_key = ""


while True :
  response = openai.Completion.create(
    model="text-davinci-003",
      prompt = input("Search : "),
      temperature=0.9,
      max_tokens=150,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.6,      
  )
  text = response["choices"][0]["text"]
  print(text)


