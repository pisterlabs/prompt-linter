import pandas as pd
df = pd.read_csv('SDW2023.csv')
user_ids = df['UserID'].tolist()
print(user_ids)

sdw2023_api_url = 'https://sdw-2023-prd.up.railway.app'

import requests
import json

def get_user(id):
    response = requests.get(f'{sdw2023_api_url}/users/{id}')
    return response.json()if response.status_code == 200 else None

users = [user for id in user_ids if (user := get_user(id)) is not None]
print(json.dumps(users, indent=2))

openai_api_key = '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'

import openai

openai.api_key = openai_api_key

def generate_ai_news(user):
     completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {
          "role": "system",
          "content": "Você é especilista em investimentos e marketing bancário."
      },
      {
          "role": "user",
          "content": f"Crie uma mensagem para {user['name']} sobre a importância de investir (máximo de 200 caracteres)"
      }
    ]
  )
     return completion.choices[0].message


for user in users:
  news = generate_ai_news(user)
  print(news)
  user['user']

def update_user(user):
    response = requests.put(f"{sdw2023_api_url}/users/{'id'}", json=user)
    return True if response.status_code == 200 else False

for user in users:
    sucess = update_user(user)
    print(f"User{user['name']} updated? {sucess}!")