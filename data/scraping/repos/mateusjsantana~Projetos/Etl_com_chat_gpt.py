import pandas as pd
import requests
import json
import openai

df = pd.read_json('https://fakerapi.it/api/v1/users?_quantity=5')
id = df


def get_user(id):
    response = requests.get('https://fakerapi.it/api/v1/users?_quantity=5')
    return response.json()['data']

users = get_user(id)

firstnames = [user['firstname'] for user in users]

print(json.dumps(firstnames, indent=2))


openai_api_key = 'KEY'
openai.api_key = openai_api_key

def generate_ai_news(firstname):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "Você é um especialista em comunicação de ganhadores de sorteios."
            },
            {"role": "user",
             "content": f"Crie emails com base nos dados de {firstname} dizendo que com base nas informações eles ganharam"
            },
        ]
    )
    return completion.choices[0].message.content.strip('\"')
    
for firstname in firstnames:
    news = generate_ai_news(firstname)
    print(news)

def update_user(user):
  response = requests.put('https://fakerapi.it/api/v1/users?_quantity=5', json=user)
  return True if response.status_code == 200 else False

for user in users:
    success = update_user(user)
    print(f"User:{user['firstname']} update? {success}")