# -*- coding: utf-8 -*-
"""sdw_dio

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Cqrtc8qM3xXohw19V77vJm4FDlW53wow
"""

pip install openai

#extrair ids
import pandas as pd

df = pd.read_csv("SDW2023.csv")
user_ids = df["UserID"].tolist()
print(user_ids)

# Utilize sua própria URL se quiser ;)
# Repositório da API: https://github.com/digitalinnovationone/santander-dev-week-2023-api
sdw2023_api_url = 'https://sdw-2023-prd.up.railway.app'

import requests
import json

def get_user(id):
  response = requests.get(f"{sdw2023_api_url}/users/{id}")

  return response.json() if response.status_code == 200 else None

#vini id 1080
users = [user for id in user_ids if (user := get_user(id)) is not None]
print(json.dumps(users, indent=2))

"""

transform - api gpt4
"""

# Documentação Oficial da API OpenAI: https://platform.openai.com/docs/api-reference/introduction
# Informações sobre o Período Gratuito: https://help.openai.com/en/articles/4936830

# Para gerar uma API Key:
# 1. Crie uma conta na OpenAI
# 2. Acesse a seção "API Keys"
# 3. Clique em "Create API Key"
# Link direto: https://platform.openai.com/account/api-keys

# Substitua o texto TODO por sua API Key da OpenAI, ela será salva como uma variável de ambiente.
openai_api_key = 'todo'

from openai.api_resources import completion
import openai
openai.api_key = openai_api_key

def generate_ai_news(user):
  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system",
     "content": "Voce é um especialista em finanças."},

    {"role": "user",
     "content": f"crie uma mensagem para {user['name']} sobre a importancia dos investimentos (maximo de 100 caracteres)"}
  ]
)

  return  completion.choices[0].message.content.strip()

for user in users:
  news = generate_ai_news(user)
  print(news)
