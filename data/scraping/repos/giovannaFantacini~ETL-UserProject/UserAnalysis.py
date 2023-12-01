# Extract

import requests
import pandas as pd
import json
import openai
import time

urlreq = 'https://64e2780eab003735881904fb.mockapi.io/apiInfo/userInfo'

def getUsers():
    response = requests.get(urlreq)
    return response.json() if response.status_code == 200 else None

users = getUsers()
print(json.dumps(users, indent=2))

# Transform

openai_api_key = ''
openai.api_key = openai_api_key

def generate_ai_message(user):
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {
          "role": "system",
          "content": "Você é um especialista em markting."
      },
      {
          "role": "user",
          "content": f"Crie uma mensagem de marketing para {user['name']} de acordo com o seu trabalho {user['job']} e o produto que ele está buscando {user['product']} (máximo de 100 caracteres)"
      }
    ]
  )
  return completion.choices[0].message.content.strip('\"')

for user in users:
  message = generate_ai_message(user)
  print(message)
  user['marketing'].update({'message': message})
  time.sleep(20)


# Load 

def update_user(user):
  response = requests.put(f"{urlreq}/{user['id']}", json=user)
  return True if response.status_code == 200 else False

for user in users:
  success = update_user(user)
  print(f"User {user['name']} updated? {success}!")
