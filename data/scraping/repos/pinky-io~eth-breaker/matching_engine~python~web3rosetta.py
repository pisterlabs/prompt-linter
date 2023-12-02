import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import json
import csv
import os
import requests


with open('dapps_dictionary4.json', 'r') as file:
    valuable_dict = json.load(file)

csv_path = '../contractsData.csv'
df_user = pd.read_csv(csv_path)

terms_list = df_user.contractName.tolist()

translation_dict = {
    "GhoToken": "Aave",
    "RocketTokenRETH": "Rocket Pool",
    "AaveGovernanceV2": "Aave",
    "ETHRegistrarController": "ENS",
    "UniversalRouter": "Uniswap",
    "PublicResolver": "ENS",
    "InitializableAdminUpgradeabilityProxy": "Aave",
    "L1ChugSplashProxy": "Base Bridge",
    "KittyCore":"CryptoKitties"
}

translated_values = {}

for term in terms_list:
    key_part = term.split()[0]
    if key_part in translation_dict:
        valuable_key = translation_dict[key_part]
        translated_values[term] = valuable_dict.get(valuable_key, "Not found")

with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(translated_values)
    for key, value in valuable_dict.items():
        writer.writerow([key, value])

terms_list.reverse()

texts_to_combine = []
for name in terms_list:
    if name in translated_values:
        texts_to_combine.append(translated_values[name])

combined_text = ' '.join(texts_to_combine)

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "user", "content": f"Summarize the following text into a concise summary of 200 characters, using small sentences talking a bit about everything and focusing more on the initial information and less on the details towards the end: {combined_text}"}
    ]
)

csv_path_wallet = '../currentwallet.csv'
df_user = pd.read_csv(csv_path_wallet)
df_user['summary'] = completion.choices[0].message.content

df_user.to_csv('./similaritydata.csv')

url = 'https://enhanced-mastiff-99.hasura.app/api/rest/update-user'

payload = {
    'wallet_address': df_user['wallet'].iloc[0],
    'summary': df_user['summary'].iloc[0]
}

load_dotenv()
api_key = os.getenv('x-hasura-admin-secret')
headers = {
    'Content-Type': 'application/json',
    'x-hasura-admin-secret': api_key
}

try:
    response = requests.patch(url, json=payload, headers=headers)
    print(response.text)
except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")