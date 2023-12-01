""" openai test - extraction info about functions or positions from bio """

import os
from pathlib import Path
from dotenv import load_dotenv
import openai


env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

#OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

# dane z pliku tekstowego
file_data = Path("..") / "data" / "sedlaczek_edward_jozef.txt"
with open(file_data, 'r', encoding='utf-8') as f:
    data = f.read()

# prompt - zapytanie do GPT, jeżeli potrzebne to z przykładami odpowiedzi
prompt = "From this text, extract information about the offices, " \
         "functions and positions held by the person Edward Józef Sedlaczek, " \
         "present them in the form of a list:\n\n" + data

response = openai.Completion.create(
  model="text-davinci-003", # najlepszy ale i najdroższy model w openai
  prompt=prompt,
  temperature=0.0, # domyślnie 0.5, zmniejszenie powoduje mniejszą 'płynność'
                   # generowanej odpowiedzi, ale jest bardziej konkretna, mniej losowa
  max_tokens=500,
  top_p=1.0,
  frequency_penalty=0.8,
  presence_penalty=0.0
)

print(response['choices'][0]['text'])

file_output = Path("..") / "output" / "sedlaczek_urzedy_temp_0_fp_0.txt"
with open(file_output, 'w', encoding='utf-8') as f:
    f.write(response['choices'][0]['text'])
