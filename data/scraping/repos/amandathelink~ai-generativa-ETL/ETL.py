sdw2023_api_url = 'https://sdw-2023-prd.up.railway.app'
import requests
import json
import pandas as pd
import openai
import os

#TODO Extrair os IDs do arquivo CSV. 
df = pd.read_csv('SWD.csv')
user_ids = df['UserID'].tolist()
#print(user_ids)

#TODO Obter os dados de cada ID usando a API da Santander Dev Week 2023
def get_user(id):
    response = requests.get(f'{sdw2023_api_url}/users/{id}')
    return response.json() if response.status_code == 200 else None

users = [user for id in user_ids if (user := get_user(id)) is not None]
#print(json.dumps(user, indent=2))


with open('key.txt') as f:
    openai_api_key = f.read()

openai.api_key = openai_api_key

def generate_ai_news(user):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
       {"role": "system", "content": "Você é um especialista em marketing bancário"},
       {"role": "user", "content": f"Crie uma mensagem para {user['name']} sobre investimentos (máximo 280 caracteres):"},
   ]
)
    return completion.choices[0].message.content.strip('\"')


for user in users:
   news = generate_ai_news(user)
   #print(news)
   user['news'].append({
    "icon": "https://digitalinnovationone.github.io/santander-dev-week-2023-api/icons/insurance.svg",
    "description":news + " " + user['name']})

#TODO Atualizar os dados de cada ID usando a API da Santander Dev Week 2023

def update_user(user):
    response = requests.put(f"{sdw2023_api_url}/users/{user['id']}", json=user)
    return True if response.status_code == 200 else False

for user in users:
    success = update_user(user)
    print(f"User {user['name']} updated successfully? {success}!")