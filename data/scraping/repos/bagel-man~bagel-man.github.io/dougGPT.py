import openai_secret_manager
import open ai
from flask import Flask, request, jsonify
from dotenv import dotenv_values

env_data = dotenv_values('.env')

secrets = openai_secrets_manager.get_secret("chatgpt")
openai.api_key = secrets[""]

app = Flask(_name_)

model = "text-davinci-003"
temperature = 0.7
max_tokens = 3500

def chat():
  user_message = request.json['message']
  
  response = openai.Completion.create(
    engine = "text-davinci-003",
    prompt = user_message,
    tempature = 0.7,
    max_tokens = 3500
  )
    
response_text = response["choices"][0]["text"].strip()
return json(response_text)

if _name_ == '_main_':
  app.run(debug = True)
