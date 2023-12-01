import pyttsx3
import os
import openai
engine = pyttsx3.init()

# your api key
openai.api_key ="API_KEY" #your api key
c = input("User: ")
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "assistant", "content": c}
  ]
)

print(completion.choices[0].message.content)

engine.say(completion.choices[0].message.content)
engine.runAndWait()
