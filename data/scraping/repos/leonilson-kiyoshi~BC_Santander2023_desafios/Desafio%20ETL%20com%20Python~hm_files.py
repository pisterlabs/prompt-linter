import pandas as pd
import numpy as np
import openai

# Função de Importação dos arquivos

def importar(arquivo, quantidade):
    dataframes = []

    for i in range(1, quantidade + 1):
        nome_arquivo = f'{arquivo}{i}.xlsx'
        if i == 1:

            try:
                df = pd.read_excel(nome_arquivo)
                df.columns = ["Devedor", "Classificação", 'Credor', 'Moeda', 'Valor']
                df = df.drop(0)
                dataframes.append(df)
                print(f'Arquivo {nome_arquivo} importado com sucesso.')
            except FileNotFoundError:
                print(f'Arquivo {nome_arquivo} não encontrado.')
        else:
            try:
                df = pd.read_excel(nome_arquivo)
                dataframes.append(df)
                print(f'Arquivo {nome_arquivo} importado com sucesso.')
            except FileNotFoundError:
                print(f'Arquivo {nome_arquivo} não encontrado.')
    return dataframes

def generate_ai_news(user):
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {
          "role": "system",
          "content": "Você é um psicólogo e precisa motivar pessoas que perderam muito dinheiro."
      },
      {
          "role": "user",
          "content": f"Crie uma mensagem motivacional para o cliente {user} que acabou de perder dinheiro com a Hotmilhas (máximo de 200 caracteres e deve conter o nome do cliente)"
      }
    ]
  )
  return completion.choices[0].message.content.strip('\"')

def salvar(nome_arquivo):
    print("salvando ...")
    arquivo = f'{nome_arquivo}.xlsx'
    df.to_excel(arquivo, index = False)
    return print(f'DataFrame salvo em {arquivo}')

# Extract
df = pd.concat(importar("ARTV3 Quirografário_Parte", 7), ignore_index=True)

# Transform

df['Valor'] = df['Valor'].replace('-', np.nan)
df['Valor'] = df['Valor'].astype(float)
Soma = df['Valor'].sum()
print(f'O valor total devido para credores Quirográficos é de R${Soma}')

openai_api_key = "[Digite a API_Key]"

openai.api_key = openai_api_key

mensagens = []
for i in range(200):
    user = df['Credor'].iloc[i]
    mensagem_gerada = generate_ai_news(user)
    mensagens.append(mensagem_gerada)

for i in range(200, len(df)):
    mensagem_gerada = "Aguarde ... menssagem será gerada em breve"
    mensagens.append(mensagem_gerada)

df['Menssagem'] = mensagens

#Load
salvar("ARTV3 Quirografário_Com_Mensagens")

